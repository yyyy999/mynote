# TileLang-Ascend NPU JIT编译流程详解

## 目录

1. [概述](#1-概述)
2. [整体架构](#2-整体架构)
3. [核心组件详解](#3-核心组件详解)
   - 3.1 [JIT装饰器入口](#31-jit装饰器入口)
   - 3.2 [compiler_npu编译器类](#32-compiler_npu编译器类)
   - 3.3 [JitKernel_NPU运行时类](#33-jitkernel_npu运行时类)
4. [编译流程详解](#4-编译流程详解)
   - 4.1 [前端处理阶段](#41-前端处理阶段)
   - 4.2 [IR转换与优化阶段](#42-ir转换与优化阶段)
   - 4.3 [NPU代码生成阶段](#43-npu代码生成阶段)
   - 4.4 [包装器生成阶段](#44-包装器生成阶段)
5. [运行时执行流程](#5-运行时执行流程)
6. [关键源码解析](#6-关键源码解析)
7. [开发者指南](#7-开发者指南)

---

## 1. 概述

TileLang-Ascend是针对华为Ascend NPU硬件优化的张量计算框架。其JIT（Just-In-Time）编译流程将用户定义的高层张量计算程序转换为可在NPU上高效执行的二进制代码。整个编译流程基于TVM（Tensor Virtual Machine）的编译基础设施，并针对Ascend NPU的特殊硬件架构进行了深度定制。

### 1.1 编译流程概览

```
用户代码 (Python装饰器)
    ↓
TIR (Tensor Intermediate Representation)
    ↓
MLIR (Multi-Level Intermediate Representation)
    ↓
NPU IR (Ascend专用中间表示)
    ↓
二进制代码 (.o文件)
    ↓
运行时加载与执行
```

### 1.2 核心文件位置

| 文件 | 路径 | 功能描述 |
|------|------|----------|
| `jit_npu.py` | `tilelang/jit/jit_npu.py` | NPU JIT编译核心实现 |
| `lower.py` | `tilelang/engine/lower.py` | TIR到MLIR的转换 |
| `phase.py` | `tilelang/engine/phase.py` | 编译Pass调度 |
| `npu_utils.py` | `tilelang/utils/npu_utils.py` | NPU工具函数 |
| `npu_utils.cpp` | `tilelang/utils/npu_utils.cpp` | NPU底层C++接口 |

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户层 (Python)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  @tilelang.jit(target="npuir")                           │  │
│  │  def my_kernel(...):                                      │  │
│  │      @T.prim_func                                         │  │
│  │      def main(...):                                       │  │
│  │          # TileLang DSL代码                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      JIT编译层 (tilelang/jit)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   jit/__init__ │→ │  compiler_npu  │→ │ JitKernel_NPU  │   │
│  │   (装饰器入口)  │  │  (编译协调器)   │  │  (运行时封装)   │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    IR转换层 (tilelang/engine)                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │    lower.py    │→ │   phase.py     │→ │  tladapter/    │   │
│  │  (IR Lowering) │  │ (Pass调度)      │  │ (MLIR转换)     │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   代码生成层 (C++ Backend)                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │ codegen_npuir  │→ │ bishengir-     │→ │   kernel.o     │   │
│  │  (NPU IR生成)  │  │ compile        │  │  (二进制)       │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    运行时层 (C++ Runtime)                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │  npu_utils.cpp │→ │ Ascend Runtime │→ │   NPU硬件      │   │
│  │  (内核加载)     │  │    API         │  │   执行         │   │
│  └────────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PrimFunc   │ ──→ │   MLIR      │ ──→ │   NPU IR    │
│  (TIR函数)  │     │  (中间表示) │     │  (Ascend)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ↓
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Launcher   │ ←── │  Wrapper    │ ←── │  Binary     │
│  (.so)      │     │  (C++代码)  │     │  (.o)       │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 3. 核心组件详解

### 3.1 JIT装饰器入口

JIT装饰器是用户与编译系统的交互入口，位于 `tilelang/jit/__init__.py`。

#### 3.1.1 装饰器定义

```python
def jit(
    func: Union[Callable[_P, _RProg], PrimFunc, None] = None,
    *,
    out_idx: Any = None,
    target: Union[str, Target] = "auto",
    target_host: Union[str, Target] = None,
    execution_backend: Literal["dlpack", "ctypes", "cython"] = "cython",
    verbose: bool = False,
    pass_configs: Optional[Dict[str, Any]] = None,
    debug_root_path: Optional[str] = None
):
```

**参数说明：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `func` | Callable/PrimFunc | None | 被装饰的函数或PrimFunc |
| `out_idx` | int/List[int] | None | 输出张量的索引位置 |
| `target` | str/Target | "auto" | 编译目标，NPU为"npuir" |
| `target_host` | str/Target | None | 主机端编译目标 |
| `execution_backend` | str | "cython" | 执行后端 |
| `verbose` | bool | False | 是否输出详细日志 |
| `pass_configs` | Dict | None | 编译Pass配置 |
| `debug_root_path` | str | None | 调试输出路径 |

#### 3.1.2 编译触发流程

当用户调用被装饰的函数时，会触发以下流程：

```python
@functools.wraps(func)
def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> Any:
    # 1. 分离调优参数
    tune_params = kwargs.pop('__tune_params', {})
    
    # 2. 构建缓存键
    key = (args, tuple(sorted(kwargs.items())))
    
    # 3. 检查缓存
    if key not in self._kernel_cache:
        # 4. 生成PrimFunc
        if callable(program_result_source):
            program_result = program_result_source(*args, **kwargs, **tune_params)
        
        # 5. 编译
        kernel_result = compile(
            program_result,
            out_idx=self.out_idx,
            target=self.target,
            ...
        )
        
        # 6. 缓存结果
        self._kernel_cache[key] = kernel_result
    
    return self._kernel_cache[key]
```

#### 3.1.3 compile函数

`compile`函数是编译的核心调度器：

```python
def compile(
    func: PrimFunc = None,
    out_idx: Union[List[int], int, None] = None,
    target: Union[str, Target] = "auto",
    ...
) -> JITKernel:
    if target == 'npuir':
        # NPU专用编译路径
        compile_npuir = compiler_npu()
        return compile_npuir.compile(func, out_idx)
    # 其他目标的编译路径
    return cached(...)
```

### 3.2 compiler_npu编译器类

`compiler_npu`类是NPU编译的核心协调器，负责管理整个编译流程。

#### 3.2.1 类结构

```python
class compiler_npu:
    def __init__(self) -> None:
        pass
    
    def compile(self, mod: PrimFunc, out_idx=None) -> JitKernel_NPU:
        # 编译主流程
        ...
```

#### 3.2.2 编译主流程

`compile`方法的完整流程如下：

```python
def compile(self, mod: PrimFunc, out_idx=None) -> JitKernel_NPU:
    # 步骤1: 保存原始模块
    self.original_mod = mod
    
    # 步骤2: 提取参数信息
    param_info = self._extract_param_info(mod, out_idx)
    
    # 步骤3: 处理负索引输出
    if out_idx is not None:
        total_params = len(param_info)
        out_idx = [i if i >= 0 else total_params + i for i in out_idx]
    
    # 步骤4: 初始化元数据
    self.metadata = {}
    self.metadata["out_idx"] = out_idx
    self.metadata["param_info"] = param_info
    
    # 步骤5: 符号变量提升
    self.mod, self.metadata["symbolic"] = _symbolic_var_promoter_pass(mod)
    
    # 步骤6: 检查调试操作
    self.need_debug = self.check_debug_op(self.mod)
    
    # 步骤7: 解析网格信息
    self._parse_grid()
    self.metadata["params"] = self.mod.params
    self.out_idx = out_idx
    self.metadata["out_idx"] = self.out_idx
    
    # 步骤8: TIR到MLIR转换
    mlir_path = lower(self.mod)
    if mlir_path.endswith(".mlir"):
        self.mlir_content = self._read_mlir_file(mlir_path)
    else:
        self.mlir_content = mlir_path
    
    # 步骤9: 解析签名信息
    self.constants = {}
    self.signature = self._parse_signature()
    
    # 步骤10: 更新元数据
    self.metadata["signature"] = self.signature
    self.metadata["primfunc"] = self.mod
    self.metadata["mlir_content"] = self.mlir_content
    
    # 步骤11: 解析NPU IR元数据
    self.lock_num = -1
    self.lock_ini_val = 0
    self._parse_npuir_metadata()
    
    # 步骤12: NPU IR编译为二进制
    self.metadata["kernel_src"] = self._npuir_to_bin_enable_npu_compile()
    
    # 步骤13: 生成包装器代码
    self.header_path = get_npu_launcher_header()
    self.wrapper_src = generate_npu_wrapper_src(
        self.constants,
        self.signature,
        self.workspace_size,
        self.metadata["mix_mode"],
        self.lock_num,
        self.lock_ini_val,
        self.need_debug,
    )
    
    # 步骤14: 构建启动器
    self.so_launcher_path = self.make_npu_launcher_stub(
        self.metadata["kernel_name"], self.header_path, self.wrapper_src
    )
    
    # 步骤15: 返回JIT Kernel对象
    return JitKernel_NPU(metadata=self.metadata, out_idx=out_idx)
```

### 3.3 JitKernel_NPU运行时类

`JitKernel_NPU`类封装了编译后的内核，提供执行接口。

#### 3.3.1 类初始化

```python
class JitKernel_NPU:
    def __init__(self, metadata: dict, out_idx=None) -> None:
        # 参数信息
        self.params = metadata["params"]
        self.signature = metadata.get("signature", {})
        self.out_idx = out_idx
        self.param_info = metadata.get("param_info", [])
        
        # 启动器路径
        self.so_launcher_path = f"{metadata['kernel_name']}.so"
        self.so_utils_path = "npu_utils.so"
        
        # 内核信息
        self.utils_kernel_src = metadata["kernel_src"]
        self.utils_shared = metadata["shared"]
        self.mlir_content = metadata["mlir_content"]
        self.mix_mode = metadata["mix_mode"]
        
        # 设备信息
        self.utils_device = torch.npu.current_device()
        self.launch_stream = torch.npu.current_stream(
            torch.npu.current_device()
        ).npu_stream
        
        # 元数据
        self.launch_packedMetadata = {
            "kernel_name": f"{metadata['name']}",
            "tensor_kinds": metadata["tensor_kinds"],
        }
        self.kernel_name = f"{metadata['name']}"
        self.tensor_kinds = metadata["tensor_kinds"]
        
        # 网格和符号信息
        self.gridfunc = metadata["gridfunc"]
        self.symbolic = metadata["symbolic"]
        self.prim_func = metadata["primfunc"]
        self.out_idx = metadata["out_idx"]
        
        # 加载启动器
        self._launch()
```

#### 3.3.2 内核执行

`__call__`方法实现了内核的执行逻辑：

```python
def __call__(self, *args: Any) -> Any:
    # 步骤1: 计算输入参数数量
    total_params = len(self.param_info)
    num_inputs = total_params - (
        len(self.out_idx) if self.out_idx is not None else 0
    )
    
    # 步骤2: 验证参数数量
    if len(args) != num_inputs:
        raise ValueError(f"Expected {num_inputs} inputs, got {len(args)}")
    
    # 步骤3: 构建输入参数映射
    orig_to_input = {}
    input_pos = 0
    for i, info in enumerate(self.param_info):
        if not info["is_output"]:
            orig_to_input[i] = input_pos
            input_pos += 1
    
    # 步骤4: 计算网格维度和动态值
    dynamic_val = self._calcu_grid(orig_to_input, *args)
    
    # 步骤5: 构建完整参数列表
    full_args = [None] * total_params
    input_ptr = 0
    
    for i, info in enumerate(self.param_info):
        if info["is_output"]:
            # 创建输出张量
            dtype = info["dtype"]
            shape = []
            for dim in info["shape"]:
                if isinstance(dim, tir.Var):
                    val = dynamic_val.get(str(dim))
                    shape.append(val)
                else:
                    shape.append(int(dim))
            device = args[0].device if args else torch.device("cpu")
            full_args[i] = torch.empty(shape, dtype=dtype, device=device)
        else:
            # 输入参数
            full_args[i] = args[input_ptr]
            input_ptr += 1
    
    # 步骤6: 添加额外参数（符号变量值）
    full_args.extend(self.extra_args)
    
    # 步骤7: 加载内核二进制
    npu_utils = NPUUtils.get()
    t_module, t_function, t_n_regs, t_n_spills = npu_utils.load_binary(
        self.utils_name,
        self.utils_kernel_src,
        self.utils_shared,
        self.utils_device,
        self.mix_mode,
    )
    
    # 步骤8: 启动内核
    self.launch_npu(
        self.launch_grid[0],
        self.launch_grid[1],
        self.launch_grid[2],
        self.launch_stream,
        t_function,
        self.launch_packedMetadata,
        self.launch_metadata,
        self.launch_enter_hook,
        self.launch_exit_hook,
        *full_args,
    )
    
    # 步骤9: 返回输出
    if self.out_idx is None:
        return None
    if len(self.out_idx) == 1:
        return full_args[self.out_idx[0]]
    else:
        return [full_args[i] for i in self.out_idx]
```

---

## 4. 编译流程详解

### 4.1 前端处理阶段

#### 4.1.1 参数信息提取

`_extract_param_info`方法从PrimFunc中提取参数信息：

```python
def _extract_param_info(self, func: PrimFunc, out_idx):
    """
    从PrimFunc中提取参数信息。
    
    返回一个字典列表，每个字典包含：
        - dtype: torch.dtype类型
        - shape: 维度列表（可能包含tir.Var用于动态形状）
        - is_output: 布尔值，指示是否为输出张量
    """
    buffer_map = func.buffer_map
    params = func.params
    info_list = []
    
    # 转换out_idx为正索引
    total_params = len(params)
    pos_out_idx = None
    if out_idx is not None:
        pos_out_idx = {i if i >= 0 else total_params + i for i in out_idx}
    
    for i, param in enumerate(params):
        is_output = pos_out_idx is not None and i in pos_out_idx
        if param in buffer_map:
            # 张量参数（有buffer）
            buffer = buffer_map[param]
            dtype_str = str(buffer.dtype)
            torch_dtype = self._tvm_dtype_to_torch(dtype_str)
            shape_expr = list(buffer.shape)
            
            # 检查零维张量（不支持作为输出）
            if is_output and len(shape_expr) == 0:
                raise ValueError(
                    f"Output parameter at index {i} has zero-dimensional shape. "
                    f"TileLang does not support scalar outputs."
                )
            
            info_list.append({
                "dtype": torch_dtype,
                "shape": shape_expr,
                "is_output": is_output,
            })
        else:
            # 标量参数
            if is_output:
                raise ValueError(
                    f"Parameter at index {i} is a scalar but marked as output."
                )
            dtype_str = str(param.dtype)
            torch_dtype = self._tvm_dtype_to_torch(dtype_str)
            info_list.append({
                "dtype": torch_dtype,
                "shape": [],  # 标量形状为空
                "is_output": False,
            })
    
    return info_list
```

#### 4.1.2 符号变量提升

`_symbolic_var_promoter_pass`函数处理动态形状中的符号变量：

```python
def _symbolic_var_promoter_pass(func: PrimFunc):
    """
    符号变量提升Pass。
    
    将buffer shape中的符号变量提升为函数参数，
    使得动态形状可以在运行时确定。
    """
    # 收集动态符号变量
    dynamic_symbolic_map = _process_dynamic_symbolic(func)
    symbolic_vars = list(dynamic_symbolic_map.keys())
    
    if len(symbolic_vars) == 0:
        return func, {}
    
    # 创建新的参数列表：原始参数 + 符号变量
    new_params = list(func.params) + symbolic_vars
    
    # 转换函数体，移除符号变量定义
    new_body = _transform_stmt(func.body, symbolic_vars)
    
    # 创建新的PrimFunc
    new_primfunc = tir.PrimFunc(
        params=new_params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
        span=func.span,
    )
    
    return new_primfunc, dynamic_symbolic_map


def _process_dynamic_symbolic(func):
    """
    收集buffer shape中使用的所有符号变量。
    
    返回: {tir.Var: (param_index, shape_dim_index)}
    """
    params = func.params
    buffer_map = func.buffer_map
    dynamic_symbolic_map = {}
    
    for i, param in enumerate(params):
        if param not in buffer_map:
            continue
        buffer = buffer_map[param]
        for j, shape in enumerate(buffer.shape):
            if isinstance(shape, tir.Var) and (shape not in dynamic_symbolic_map):
                dynamic_symbolic_map[shape] = (i, j)
    
    return dynamic_symbolic_map
```

#### 4.1.3 网格信息解析

`_parse_grid`方法提取并行执行的网格维度：

```python
def _parse_grid(self):
    """
    从PrimFunc中提取blockIdx.x的值作为网格维度。
    """
    launcher = LaunchThreadExtractor()
    expr = launcher.extract(self.mod, "blockIdx.x")
    self.metadata["gridfunc"] = str(expr)


class LaunchThreadExtractor:
    """
    从PrimFunc中提取线程范围表达式的访问器。
    """
    def __init__(self) -> None:
        self.expressions = []
    
    def visit_thread_extent(self, node):
        # 递归访问AST节点
        if hasattr(node, "body"):
            self.visit_thread_extent(node.body)
        if hasattr(node, "then_case"):
            self.visit_thread_extent(node.then_case)
        if hasattr(node, "else_case"):
            self.visit_thread_extent(node.else_case)
        if hasattr(node, "block"):
            self.visit_thread_extent(node.block)
        
        # 检查是否为目标线程范围
        if (
            hasattr(node, "attr_key")
            and node.attr_key == "thread_extent"
            and node.node.thread_tag == self.thread
        ):
            self.expressions.append(node.value)
    
    def extract(self, node: PrimFunc, thread: str):
        self.thread = thread
        self.visit_thread_extent(node)
        if self.expressions is None:
            return None
        return self.expressions[0]
```

### 4.2 IR转换与优化阶段

#### 4.2.1 Lower函数

`lower`函数（位于`tilelang/engine/lower.py`）负责TIR到MLIR的转换：

```python
def lower(
    func_or_mod: Union[tir.PrimFunc, tvm.IRModule],
    target: Union[str, Target] = "auto",
    target_host: Optional[Union[str, Target]] = None,
    runtime_only=False,
    enable_host_codegen=False,
    enable_device_compile=False,
) -> CompiledArtifact:
    """
    将TIR函数转换为MLIR。
    
    参数:
        func_or_mod: TIR函数或IR模块
        target: 编译目标（"npuir"用于NPU）
        target_host: 主机端目标
        runtime_only: 是否仅生成运行时代码
        enable_host_codegen: 是否启用主机代码生成
        enable_device_compile: 是否启用设备代码编译
    
    返回:
        CompiledArtifact对象，包含生成的代码
    """
    mod = func_or_mod
    params = None
    
    # 处理单个PrimFunc
    if isinstance(func_or_mod, tir.PrimFunc):
        func = func_or_mod
        params = extrac_params(func) if not runtime_only else None
        mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    
    # 确定编译目标
    if isinstance(target, str):
        target = determine_target(target)
    
    target_host = canon_target_host(target, target_host)
    target_host = tvm.target.Target.canon_target(target_host)
    target = tvm.target.Target(target, target_host)
    
    # 阶段1: Lower和合法化IR
    mod = LowerAndLegalize(mod, target)
    
    # 阶段2: 针对目标优化IR
    mod = OptimizeForTarget(mod, target)
    
    # NPU特定处理
    if target.kind.name == "npuir":
        # 设备代码生成
        codegen_mod = device_codegen(mod, target)
        mlir_str = codegen_mod.get_source()
        
        # 应用tladapter passes
        tladapter_passes = [
            transforms.mlir.canonicalize(top_down=True),
            transforms.bishengir.adapt_triton_kernel,
        ]
        for i, p in enumerate(tladapter_passes):
            mlir_str = p(mlir_str)
        
        return mlir_str
    
    # 其他目标的处理...
    ...
```

#### 4.2.2 LowerAndLegalize阶段

`LowerAndLegalize`函数执行IR的初始转换和合法化：

```python
def LowerAndLegalize(mod: IRModule, target: Target) -> IRModule:
    """
    IR Lower和合法化阶段。
    
    对于NPU目标，执行简化的处理流程。
    """
    # 绑定目标设备信息
    mod = tir.transform.BindTarget(target)(mod)
    
    if target.kind.name == "npuir":
        # NPU特定路径
        mod = tir.transform.Simplify()(mod)
        mod = tir.transform.RemoveNoOp()(mod)
        return mod
    
    # 通用路径
    # 合法化前端IR
    mod = tilelang.transform.FrontendLegalize()(mod)
    # 简化IR表达式
    mod = tir.transform.Simplify()(mod)
    # 推断内存布局
    mod = tilelang.transform.LayoutInference()(mod)
    # 降低高层tile操作
    mod = tilelang.transform.LowerTileOp()(mod)
    # 合法化向量化循环
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # 添加内存访问安全检查
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # 再次简化
    mod = tir.transform.Simplify()(mod)
    # 动态形状循环向量化
    mod = tilelang.transform.LoopVectorizeDynamic()(mod)
    
    return mod
```

#### 4.2.3 OptimizeForTarget阶段

`OptimizeForTarget`函数针对特定目标进行优化：

```python
def OptimizeForTarget(mod: IRModule, target: Target) -> IRModule:
    """
    针对特定目标优化IR。
    
    对于NPU，执行NPU特定的优化Pass。
    """
    pass_ctx = tilelang.transform.get_pass_context()
    
    if target.kind.name == "npuir":
        # NPU特定优化路径
        
        # NPU循环向量化
        # 位置要求：
        # 1. 必须在LowerOpaqueBlock之前，否则临时buffer无法正确变成T.decl_buffer
        # 2. 最好在PlanAndUpdateBufferAllocationLocation之前，复用其内存重用能力
        mod = tilelang.transform.NpuLoopVectorize()(mod)
        
        # 规划和更新buffer分配位置
        mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
        
        # 降低不透明块
        mod = tir.transform.LowerOpaqueBlock()(mod)
        
        # 移除空操作
        mod = tir.transform.RemoveNoOp()(mod)
        
        return mod
    
    # CUDA等其他目标的优化路径...
    ...
```

#### 4.2.4 device_codegen函数

`device_codegen`函数生成设备端代码：

```python
def device_codegen(device_mod: tvm.IRModule, target: Target) -> tvm.IRModule:
    """
    设备端代码生成。
    
    对于NPU目标，调用NPU特定的代码生成器。
    """
    if target.kind.name == "npuir":
        # 根据环境变量选择代码生成模式
        TILELANG_ASCEND_MODE = os.environ.get('TILELANG_ASCEND_MODE')
        
        if TILELANG_ASCEND_MODE is None:
            # 默认使用API模式
            device_mod = tvm._ffi.get_global_func(
                "target.build.tilelang_npuir_apis"
            )(device_mod, target)
        elif TILELANG_ASCEND_MODE.lower().strip() in ['expert', 'exp', 'e']:
            # 专家模式
            device_mod = tvm._ffi.get_global_func(
                "target.build.tilelang_npuir_apis"
            )(device_mod, target)
        else:
            # 开发模式
            device_mod = tvm._ffi.get_global_func(
                "target.build.tilelang_npuir_dev"
            )(device_mod, target)
        
        return device_mod
    
    # 其他目标的处理...
    ...
```

### 4.3 NPU代码生成阶段

#### 4.3.1 NPU IR元数据解析

`_parse_npuir_metadata`方法从生成的MLIR中提取NPU特定信息：

```python
def _parse_npuir_metadata(self) -> None:
    """
    从NPU IR中解析元数据。
    
    提取并更新以下字段：
      - mix_mode: 混合模式（mix/aic/aiv）
      - kernel_name: 内核名称
      - tensor_kinds: 张量类型列表
      - shared: 共享内存大小
      - name: 组合的内核名称
    """
    # 正则表达式定义
    # 示例: func.func @gather_sorted_kernel(%arg0: ...) -> gather_sorted_kernel
    KERNEL_NAME_REGEX = r"func\.func\s+@(\w+)"
    
    # 示例: hivm.module_core_type<MIX> -> MIX
    MIX_MODE_REGEX = r"#hivm\.module_core_type<([^>]+)>"
    
    # 示例: test_mix_aic -> test
    MIX_SUFFIX_REGEX = r"_(mix_aic|mix_aiv)$"
    
    # 设置共享内存（NPU后端暂不限制）
    self.metadata["shared"] = 1
    
    # 提取内核名称
    kernel_name = re.search(KERNEL_NAME_REGEX, self.mlir_content).group(1)
    self.metadata["kernel_name"] = kernel_name
    
    # 移除mix后缀
    self.metadata["name"] = re.sub(MIX_SUFFIX_REGEX, "", kernel_name)
    
    # 张量类型（当前硬编码为空）
    self.metadata["tensor_kinds"] = []
    
    # 提取混合模式
    self.metadata["mix_mode"] = (
        re.search(MIX_MODE_REGEX, self.mlir_content).group(1).lower()
    )
```

#### 4.3.2 签名解析

`_parse_signature`方法解析函数签名：

```python
def _parse_signature(self) -> dict:
    """
    从MLIR文本中解析参数类型。
    
    返回: {参数索引: 类型字符串}
    """
    # 定义目标数据类型
    target_types = {
        "i1", "i8", "i16", "i32", "i64",
        "u32", "u64",
        "fp16", "bf16", "fp32", "f32", "fp64", "f16",
    }
    
    # 提取函数签名部分
    pattern = r"func\.func\s*@[^(]*\(([^)]*)\)"
    match = re.search(pattern, self.mlir_content)
    
    if not match:
        return {}
    
    params_str = match.group(1)
    
    # 分割参数
    params = []
    current_param = ""
    brace_count = 0
    angle_count = 0
    
    for char in params_str:
        if char == "," and brace_count == 0 and angle_count == 0:
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "<":
                angle_count += 1
            elif char == ">":
                angle_count -= 1
    
    if current_param:
        params.append(current_param.strip())
    
    result = {}
    index = 0
    
    # 跳过编译器插入的参数（前3个和后6个）
    for param in params[3:-6]:
        found_type = None
        for t_type in target_types:
            # 检查带x前缀的类型（如xf16）
            x_pattern = r"\bx" + t_type + r"\b"
            if re.search(x_pattern, param):
                found_type = "*" + t_type
                break
            # 检查普通类型
            elif re.search(r"\b" + t_type + r"\b", param):
                found_type = t_type
                break
        
        if found_type:
            # 特殊处理：f16 -> fp16, f32 -> fp32
            if found_type == "f16":
                found_type = "fp16"
            elif found_type == "*f16":
                found_type = "*fp16"
            elif found_type == "f32":
                found_type = "fp32"
            elif found_type == "*f32":
                found_type = "*fp32"
            
            result[index] = found_type
            index += 1
    
    return result
```

#### 4.3.3 NPU IR编译为二进制

`_npuir_to_bin_enable_npu_compile`方法调用NPU编译器：

```python
def _npuir_to_bin_enable_npu_compile(self):
    """
    将NPU IR编译为二进制代码。
    
    使用bishengir-compile编译器进行编译。
    """
    linalg = self.mlir_content
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入MLIR文件
        ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
        Path(ttadapter_path).write_text(linalg)
        
        # 设置输出路径
        bin_file = os.path.join(tmpdir, "kernel")
        bin_path = os.path.join(tmpdir, "kernel.o")
        so_path = os.path.join(tmpdir, "libkernel.so")
        
        # 获取编译器路径
        npu_compiler_path = get_npucompiler_path()
        
        # 构建编译选项
        _compile_option_list = [
            "--enable-auto-multi-buffer=true",
            "--enable-triton-kernel-compile=true",
            "--enable-hivm-compile=true",
        ]
        
        # 根据模式添加选项
        TILELANG_ASCEND_MODE = os.environ.get("TILELANG_ASCEND_MODE")
        if TILELANG_ASCEND_MODE is None or TILELANG_ASCEND_MODE.lower().strip() in [
            "expert", "exp", "e",
        ]:
            _compile_option_list.append("--disable-hivm-tensor-compile=true")
        
        # 构建完整命令
        cmd_list = (
            [npu_compiler_path, ttadapter_path]
            + _compile_option_list
            + ["-o", bin_file]
        )
        
        # 执行编译
        try:
            ret = subprocess.run(
                cmd_list, capture_output=True, check=True, text=True
            )
            print("AscendNPU IR compile success:", ret.stdout)
        except subprocess.CalledProcessError as e:
            # 打印IR和错误信息
            print("AscendNPU IR:\n")
            print(self.mlir_content)
            print("err cmd:", " ".join(cmd_list))
            print(f"err code: {e.returncode}")
            print("err info:", e.stderr)
            sys.exit(1)
        
        # 获取工作区大小
        result = self._get_workspace_size(
            so_path, "_infer_workspace_shape_function"
        )
        self.workspace_size = result
        
        # 验证输出文件
        if not Path(bin_path).exists():
            err_lines = [
                "AscendNPU IR compile reported success but output object was not generated.",
                f"Expected output: {bin_path}",
                f"cmd: {' '.join(cmd_list)}",
            ]
            if ret.stdout:
                err_lines.append(f"stdout:\n{ret.stdout}")
            if ret.stderr:
                err_lines.append(f"stderr:\n{ret.stderr}")
            raise RuntimeError("\n".join(err_lines))
        
        # 读取二进制内容
        return Path(bin_path).read_bytes()
```

**编译选项说明：**

| 选项 | 描述 |
|------|------|
| `--enable-auto-multi-buffer=true` | 启用自动多缓冲区优化 |
| `--enable-triton-kernel-compile=true` | 启用Triton风格内核编译模式 |
| `--enable-hivm-compile=true` | 启用HIVM编译流程 |
| `--disable-hivm-tensor-compile=true` | 禁用HIVM张量编译（专家模式） |

### 4.4 包装器生成阶段

#### 4.4.1 包装器代码生成

`generate_npu_wrapper_src`函数生成C++包装器代码：

```python
def generate_npu_wrapper_src(
    constants, signature, workspace_size, mix_mode, lock_num, lock_ini_val, need_debug
):
    """
    生成NPU内核启动器的C++包装器代码。
    
    参数:
        constants: 常量参数字典
        signature: 函数签名字典
        workspace_size: 工作区大小
        mix_mode: 混合模式（aic/aiv）
        lock_num: 锁数量
        lock_ini_val: 锁初始值
        need_debug: 是否需要调试支持
    
    返回:
        C++源代码字符串
    """
    # 类型转换函数
    def _ty_to_cpp(ty):
        if ty[0] == "*":
            return "void*"
        return {
            "i1": "int32_t",
            "i8": "int8_t",
            "i16": "int16_t",
            "i32": "int32_t",
            "i64": "int64_t",
            "u32": "uint32_t",
            "u64": "uint64_t",
            "fp16": "float",
            "bf16": "float",
            "fp32": "float",
            "fp64": "double",
        }[ty]
    
    # ... 生成代码 ...
    
    return f"""
#include "npu_launcher.h"
#define PY_SSIZE_T_CLEAN
{"#define __CCE_ENABLE_PRINT__" if need_debug else ""}
{extract_device_print_code_from_cann() if need_debug else ""}

// 设备指针处理结构
{cpp_device_pointer}

// 内核启动函数
static void _launch(
    const char* kernelName,
    const void* func,
    rtStream_t stream,
    int gridX, int gridY, int gridZ,
    std::vector<std::vector<int64_t>> &tensorShapes,
    std::vector<int> &tensorKinds,
    {arg_decls}
) {{
    // 计算block数量
    uint32_t blockNum = gridX * gridY * gridZ;
    
    // 获取FFTS地址
    void *ffts_addr = NULL;
    uint32_t ffts_len;
    ret = rtGetC2cCtrlAddr((uint64_t*)&ffts_addr, &ffts_len);
    
    // 分配同步锁
    void *syncBlockLock = NULL;
    // ... 锁分配代码 ...
    
    // 分配工作区
    void *workspace_addr = NULL;
    // ... 工作区分配代码 ...
    
    // 构建参数结构
    struct __attribute__((packed)) {{
        void* ffts_addr __attribute__((aligned(8)));
        void* syncBlockLock __attribute__((aligned(8)));
        void* workspace_addr __attribute__((aligned(8)));
        // ... 参数字段 ...
    }} args = {{ /* 初始化 */ }};
    
    // 启动内核
    ret = rtKernelLaunch(func, blockNum, static_cast<void*>(&args), sizeof(args), NULL, stream);
}}

// Python调用入口
static PyObject* launch(PyObject* self, PyObject* args) {{
    // 解析Python参数
    int gridX, gridY, gridZ;
    rtStream_t stream;
    const void *function;
    // ... 其他参数 ...
    
    if (!PyArg_ParseTuple(args, "{format}", &gridX, &gridY, &gridZ, ...)) {{
        return NULL;
    }}
    
    // 提取设备指针
    // ... 指针提取代码 ...
    
    // 调用_launch
    _launch(kernelName, function, stream, gridX, gridY, gridZ, tensorShapes, tensorKinds, ...);
    
    Py_RETURN_NONE;
}}

// Python模块定义
static PyMethodDef ModuleMethods[] = {{
    {{"launch", launch, METH_VARARGS, "Entry point for all kernels with this signature"}},
    {{NULL, NULL, 0, NULL}}
}};

static struct PyModuleDef ModuleDef = {{
    PyModuleDef_HEAD_INIT,
    "__tilelang_launcher",
    NULL,
    -1,
    ModuleMethods
}};

PyMODINIT_FUNC PyInit___tilelang_launcher(void) {{
    PyObject *m = PyModule_Create(&ModuleDef);
    if (m == NULL) {{
        return NULL;
    }}
    PyModule_AddFunctions(m, ModuleMethods);
    return m;
}}
"""
```

#### 4.4.2 启动器构建

`make_npu_launcher_stub`方法构建启动器共享库：

```python
def make_npu_launcher_stub(self, name, header_src, wrapper_src, debug=False):
    """
    生成启动器存根以启动内核。
    
    参数:
        name: 内核名称
        header_src: 头文件路径
        wrapper_src: 包装器源代码
        debug: 是否启用调试
    
    返回:
        共享库路径
    """
    # 获取预编译缓存路径
    precompile_cache_path = get_runtime_file_cache(header_src)
    header_path = os.path.join(precompile_cache_path, "npu_launcher.h")
    precompile_header_path = os.path.join(
        precompile_cache_path, "npu_launcher.h.gch"
    )
    
    # 检查预编译头是否存在
    if not (
        os.path.exists(precompile_header_path)
        and os.path.getsize(precompile_header_path) > 0
    ):
        print("Precompiling NPU launcher header...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # 复制头文件
            safe_copy(header_src, header_path)
            tmp_header_gch_path = os.path.join(tmpdir, "npu_launcher.h.gch")
            # 预编译头文件
            precompile_npu_ext(header_path, tmp_header_gch_path)
            # 复制预编译结果
            safe_copy(tmp_header_gch_path, precompile_header_path)
    
    # 编译包装器
    with tempfile.TemporaryDirectory() as tmpdir:
        dst_path = os.path.join(tmpdir, f"{name}.cxx")
        with open(dst_path, "w") as f:
            f.write(wrapper_src)
        
        # 构建共享库
        so = build_npu_ext(
            name, header_path, dst_path,
            kernel_launcher="torch", precompile=True
        )
        
        return so
```

---

## 5. 运行时执行流程

### 5.1 NPUUtils工具类

`NPUUtils`类提供了与NPU硬件交互的底层接口：

```python
class NPUUtils(object):
    """
    Ascend NPU工具的单例辅助类。
    
    首次使用时编译并加载共享库，后续调用返回同一实例。
    """
    
    _initialized = False
    
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        
        # 编译npu_utils.so
        pkg_root = os.path.dirname(os.path.abspath(__file__))
        npu_utils_cpp = os.path.join(pkg_root, "npu_utils.cpp")
        
        if os.path.exists(npu_utils_cpp):
            cache_path = get_runtime_file_cache(npu_utils_cpp)
            fname_path = os.path.join(cache_path, "npu_utils.so")
            
            if not (os.path.exists(fname_path) and os.path.getsize(fname_path) > 0):
                with tempfile.TemporaryDirectory() as tmpdir:
                    dst_path = os.path.join(tmpdir, "npu_utils.cxx")
                    safe_copy(npu_utils_cpp, dst_path)
                    so = build_npu_ext("npu_utils", None, dst_path, kernel_launcher="torch")
                    safe_copy(so, fname_path)
        
        # 加载模块
        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", str(fname_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
        self._initialized = True
    
    @classmethod
    def get(cls):
        """返回单例实例"""
        return cls()
    
    def load_binary(self, name, kernel, shared, device, mix_mode):
        """加载内核二进制"""
        return self.npu_utils_mod.load_kernel_binary(
            name, kernel, shared, device, mix_mode
        )
    
    @functools.lru_cache()
    def get_arch(self):
        """返回Ascend SoC版本"""
        return self.npu_utils_mod.get_arch()
    
    @functools.lru_cache()
    def get_aicore_num(self):
        """返回AI Core数量"""
        return self.npu_utils_mod.get_aicore_num()
    
    @functools.lru_cache()
    def get_aivector_core_num(self):
        """返回AI Vector Core数量"""
        return self.get_aicore_num() * 2
    
    @functools.lru_cache()
    def get_device_num(self):
        """返回设备数量"""
        return self.npu_utils_mod.get_device_num()
```

### 5.2 npu_utils.cpp底层实现

`npu_utils.cpp`提供了C++级别的NPU接口：

```cpp
// 内核注册函数
static std::tuple<void *, void *>
registerKernel(const char *name, const void *data, size_t data_size, int shared,
               int device, const char *kernel_mode_str) {
    rtError_t rtRet;
    
    // 设置设备二进制信息
    rtDevBinary_t devbin;
    devbin.data = data;
    devbin.length = data_size;
    
    // 根据内核模式设置magic number
    const std::string kernel_mode{kernel_mode_str};
    if (kernel_mode == "aiv")
        devbin.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    else
        devbin.magic = RT_DEV_BINARY_MAGIC_ELF;
    devbin.version = 0;
    
    // 设置设备
    rtRet = rtSetDevice(device);
    if (rtRet != RT_ERROR_NONE) {
        printf("rtSetDevice failed, 0x%x\n", rtRet);
        return {NULL, NULL};
    }
    
    // 注册设备二进制
    void *devbinHandle = NULL;
    rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
    if (rtRet != RT_ERROR_NONE) {
        printf("rtDevBinaryRegister failed, 0x%x\n", rtRet);
        return {NULL, NULL};
    }
    
    // 注册函数
    std::string stubName = name;
    stubName += "_" + std::to_string(registered_names[name]);
    registered_names[name]++;
    
    auto registered = func_stubs.emplace(stubName, std::make_unique<size_t>(0));
    void *func_stub_handle = registered.first->second.get();
    
    rtRet = rtFunctionRegister(devbinHandle, func_stub_handle, stubName.c_str(),
                               (void *)name, 0);
    if (rtRet != RT_ERROR_NONE) {
        printf("rtFunctionRegister failed, 0x%x\n", rtRet);
        return {NULL, NULL};
    }
    
    return std::make_tuple(devbinHandle, func_stub_handle);
}

// Python接口：加载内核二进制
static PyObject *loadKernelBinary(PyObject *self, PyObject *args) {
    const char *name;        // 内核名称
    const char *data;        // 二进制指针
    Py_ssize_t data_size;    // 二进制大小
    int shared;              // 共享内存
    int device;              // 设备ID
    const char *kernel_mode; // 内核模式
    
    if (!PyArg_ParseTuple(args, "ss#iis", &name, &data, &data_size, &shared,
                          &device, &kernel_mode)) {
        return NULL;
    }
    
    auto [module_handle, func_handle] =
        registerKernel(name, data, data_size, shared, device, kernel_mode);
    
    uint64_t mod = reinterpret_cast<uint64_t>(module_handle);
    uint64_t func = reinterpret_cast<uint64_t>(func_handle);
    
    return Py_BuildValue("(KKii)", mod, func, 0, 0);
}
```

### 5.3 内核启动流程

内核启动的完整流程如下：

```
Python调用 kernel(a, b, c)
    ↓
JitKernel_NPU.__call__()
    ↓
计算网格维度 (_calcu_grid)
    ↓
构建参数列表
    ↓
NPUUtils.load_binary() → loadKernelBinary()
    ↓
rtDevBinaryRegister() [注册二进制]
    ↓
rtFunctionRegister() [注册函数]
    ↓
launch_npu() → _launch()
    ↓
rtKernelLaunch() [启动内核]
    ↓
NPU硬件执行
```

---

## 6. 关键源码解析

### 6.1 动态形状处理

动态形状是JIT编译中的关键挑战。TileLang通过符号变量机制处理动态形状：

```python
# 示例：动态形状矩阵乘法
@tilelang.jit(target="npuir")
def dynamic_matmul(M, N, K, block_M, block_N):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),  # M和K是符号变量
        B: T.Tensor((K, N), "float16"),  # K和N是符号变量
        C: T.Tensor((M, N), "float16")   # M和N是符号变量
    ):
        # 内核实现
        ...
    return main

# 编译时处理
# 1. _process_dynamic_symbolic() 收集 {M: (0,0), K: (0,1), N: (1,1)}
# 2. _symbolic_var_promoter_pass() 将符号变量提升为参数
# 3. 运行时从输入张量形状推断符号变量值
```

### 6.2 网格维度计算

网格维度决定了并行执行的block数量：

```python
def _calcu_grid(self, orig_to_input, *args: Any):
    """
    根据符号变量和输入张量计算网格维度。
    """
    dynamic_val = {}
    extra_args = []
    
    for key, pos in self.symbolic.items():
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            tensor_idx, dim_idx = pos[0], pos[1]
            if tensor_idx in orig_to_input:
                pos = orig_to_input[tensor_idx]
                arg = args[pos]
                if isinstance(arg, torch.Tensor) and dim_idx < len(arg.shape):
                    value = arg.shape[dim_idx]
                    dynamic_val[str(key)] = value
                    extra_args.append(value)
    
    self.extra_args = extra_args
    
    # 替换网格函数中的符号变量
    result = replace_by_longest_key(self.gridfunc, dynamic_val)
    
    # 计算网格值
    grid_value = eval(
        result,
        {"__builtins__": {}},
        {"math": __import__("math"), **dynamic_val},
    )
    
    if hasattr(grid_value, "__iter__"):
        self.launch_grid = [int(x) for x in grid_value]
    else:
        self.launch_grid = [int(grid_value), 1, 1]
    
    return dynamic_val
```

### 6.3 工作区管理

工作区是NPU内核执行时需要的临时存储空间：

```python
def _get_workspace_size(self, lib_path, suffix, default=32768):
    """
    从编译后的内核中获取工作区大小。
    
    尝试查找并调用 infer_workspace_shape_function。
    """
    if not os.path.exists(lib_path):
        return default
    
    symbols = []
    try:
        # 使用nm命令获取符号表
        result = subprocess.run(
            ["nm", "-D", lib_path], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    sym_name = parts[2]
                    if sym_name.endswith(suffix):
                        symbols.append(sym_name)
    except (subprocess.SubprocessError, FileNotFoundError, OSError, TimeoutError):
        pass
    
    if not symbols:
        return default
    
    # 加载库并调用函数
    try:
        lib = ctypes.CDLL(lib_path)
    except OSError:
        return default
    
    for func_name in symbols:
        try:
            func = getattr(lib, func_name)
            func.restype = ctypes.c_int
            return func()
        except (AttributeError, OSError, TypeError):
            continue
    
    return default
```

### 6.4 调试支持

TileLang支持在NPU内核中打印调试信息：

```python
def check_debug_op(self, func) -> bool:
    """
    检查函数中是否包含调试操作。
    
    只有在存在调试操作时才启用设备端打印，避免性能损失。
    """
    assert isinstance(func, PrimFunc), "Expected func to be a PrimFunc"
    
    found = False
    
    def visit(node):
        nonlocal found
        if isinstance(node, tir.Call) and "debug" in node.op.name:
            found = True
    
    tir.stmt_functor.post_order_visit(func.body, visit)
    return found
```

当启用调试时，包装器代码会包含CCE打印头文件：

```python
def extract_device_print_code_from_cann():
    """
    从CANN安装目录提取设备端打印代码。
    """
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(
        ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib"
    )
    
    # 读取并处理头文件
    # ...
```

---

## 7. 开发者指南

### 7.1 环境配置

在使用TileLang-Ascend之前，需要正确配置环境：

```bash
# 设置Ascend环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置环境变量
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit

# 设置NPU编译器路径（如果不在PATH中）
export TILELANG_NPU_COMPILER_PATH=/path/to/compiler

# 启用IR转储（调试用）
export TILELANG_DUMP_IR=1

# 设置编译模式
export TILELANG_ASCEND_MODE=expert  # expert模式
# 或
export TILELANG_ASCEND_MODE=dev     # 开发模式
```

### 7.2 编写NPU内核

以下是一个完整的NPU矩阵乘法示例：

```python
import torch
import tilelang
import tilelang.language as T

@tilelang.jit(target="npuir")
def matmul(block_M, block_N, K_L1, dtype="float16", accum_dtype="float32"):
    """
    NPU矩阵乘法内核。
    
    参数:
        block_M: M方向的block大小
        block_N: N方向的block大小
        K_L1: L1缓冲区的K维度分块大小
        dtype: 输入数据类型
        accum_dtype: 累加数据类型
    """
    # 计算block数量
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype)
    ):
        # 定义NPU内核
        with T.Kernel(m_num*n_num, is_npu=True) as (cid, _):
            with T.Scope("Cube"):
                # 计算当前block的起始位置
                bx = cid // n_num * block_M
                by = cid % n_num * block_N
                
                # 分配L1缓冲区
                A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                
                # 分配L0C缓冲区（累加器）
                C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                # K维度分块循环
                for i in T.serial(T.ceildiv(K, K_L1)):
                    # 加载A矩阵块（ND到NZ格式转换）
                    T.load_nd2nz(A[bx, i * K_L1], A_BUF, [block_M, K_L1])
                    # 加载B矩阵块
                    T.load_nd2nz(B[i * K_L1, by], B_BUF, [K_L1, block_N])

                    # 执行矩阵乘法
                    if i == 0:
                        T.gemm(A_BUF, B_BUF, C_BUF, initC=True, 
                               b_transpose=False, size=[block_M, K_L1, block_N])
                    else:
                        T.gemm(A_BUF, B_BUF, C_BUF, initC=False,
                               b_transpose=False, size=[block_M, K_L1, block_N])

                    # 存储结果（NZ到ND格式转换）
                    T.store_fixpipe(C_BUF, C[bx, by],
                        size=[block_M, block_N], enable_nz2nd=True)

    return main

# 使用示例
def test_matmul():
    M, N, K = 1024, 512, 2048
    
    # 编译内核
    func = matmul(128, 256, 16)
    
    # 准备输入
    a = torch.randn(M, K).half().npu()
    b = torch.randn(K, N).half().npu()
    c = torch.randn(M, N).half().npu()
    
    # 执行内核
    func(a, b, c)
    
    # 验证结果
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
```

### 7.3 性能优化建议

1. **分块大小选择**
   - 根据L1/L0C缓冲区大小选择合适的分块
   - 考虑数据重用和内存带宽

2. **内存布局优化**
   - 使用`load_nd2nz`进行格式转换
   - 合理使用L1缓冲区减少全局内存访问

3. **流水线优化**
   - 利用双缓冲技术隐藏内存延迟
   - 合理安排计算和数据传输

4. **调试技巧**
   - 使用`TILELANG_DUMP_IR=1`查看中间IR
   - 使用`T.print()`在内核中打印调试信息
   - 使用`benchmark()`方法测量性能

### 7.4 常见问题排查

1. **编译失败**
   - 检查`ASCEND_HOME_PATH`环境变量
   - 确认`bishengir-compile`在PATH中
   - 查看错误信息中的IR内容

2. **运行时错误**
   - 检查输入张量的设备和数据类型
   - 确认动态形状参数正确传递
   - 查看NPU驱动日志

3. **性能问题**
   - 分析内核执行时间
   - 检查内存访问模式
   - 优化分块策略

---

## 附录

### A. 相关文件清单

| 文件路径 | 功能描述 |
|----------|----------|
| `tilelang/jit/__init__.py` | JIT装饰器和编译入口 |
| `tilelang/jit/jit_npu.py` | NPU JIT编译核心实现 |
| `tilelang/engine/lower.py` | TIR到MLIR转换 |
| `tilelang/engine/phase.py` | 编译Pass调度 |
| `tilelang/transform/__init__.py` | 编译Pass定义 |
| `tilelang/tladapter/transforms/` | MLIR转换Pass |
| `tilelang/utils/npu_utils.py` | NPU工具函数 |
| `tilelang/utils/npu_utils.cpp` | NPU底层C++接口 |
| `tilelang/utils/npu_launcher.h` | 启动器头文件 |

### B. 编译Pass流程图

```
PrimFunc
    │
    ├── BindTarget
    │
    ├── Simplify
    │
    ├── RemoveNoOp
    │
    ├── NpuLoopVectorize
    │
    ├── PlanAndUpdateBufferAllocationLocation
    │
    ├── LowerOpaqueBlock
    │
    └── RemoveNoOp
        │
        ↓
    MLIR (NPU IR)
```

### C. 运行时API调用流程

```
Python调用
    │
    ├── JitKernel_NPU.__call__()
    │   │
    │   ├── _calcu_grid()          # 计算网格维度
    │   │
    │   ├── NPUUtils.load_binary() # 加载内核
    │   │   │
    │   │   └── loadKernelBinary() # C++接口
    │   │       │
    │   │       ├── rtSetDevice()
    │   │       ├── rtDevBinaryRegister()
    │   │       └── rtFunctionRegister()
    │   │
    │   └── launch_npu()           # 启动内核
    │       │
    │       └── _launch()          # C++启动函数
    │           │
    │           ├── rtGetC2cCtrlAddr()
    │           ├── rtMalloc()     # 分配锁和工作区
    │           └── rtKernelLaunch() # 启动内核
    │
    └── 返回结果
```

---

本文档详细介绍了TileLang-Ascend的JIT编译流程，包括前端处理、IR转换、代码生成和运行时执行等各个环节。开发者可以通过本文档深入理解编译器的工作原理，并根据需要进行定制和优化。
