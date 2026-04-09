# TileLang-Ascend NPU JIT编译流程（开发者精简版）

## 1. 核心概念

TileLang-Ascend是基于TVM的Ascend NPU张量计算框架，通过JIT编译将Python DSL代码转换为NPU二进制。

### 编译流程概览

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

### 核心文件

| 文件                            | 功能          |
| ----------------------------- | ----------- |
| `tilelang/jit/jit_npu.py`     | NPU JIT编译核心 |
| `tilelang/engine/lower.py`    | TIR到MLIR转换  |
| `tilelang/engine/phase.py`    | 编译Pass调度    |
| `tilelang/utils/npu_utils.py` | NPU工具函数     |

***

## 2. 整体架构

```
┌─────────────────────────────────────────┐
│         用户层 (Python)                  │
│  @tilelang.jit(target="npuir")          │
│  def my_kernel(...): ...                │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       JIT编译层 (tilelang/jit)           │
│  jit装饰器 → compiler_npu → JitKernel_NPU│
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      IR转换层 (tilelang/engine)          │
│  lower.py → phase.py → tladapter/       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│     代码生成层 (C++ Backend)             │
│  codegen_npuir → bishengir-compile → .o │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      运行时层 (C++ Runtime)              │
│  npu_utils.cpp → Ascend Runtime → NPU   │
└─────────────────────────────────────────┘
```

***

## 3. 编译流程核心步骤

### 3.1 前端处理

**参数信息提取**：从PrimFunc提取张量参数的dtype、shape和is\_output信息

**符号变量提升**：将动态形状中的符号变量提升为函数参数

**网格信息解析**：提取blockIdx.x作为并行执行的网格维度

### 3.2 IR转换与优化

**LowerAndLegalize阶段**：

- 绑定目标设备
- 简化IR表达式
- 移除空操作

**OptimizeForTarget阶段**：

- NPU循环向量化
- Buffer分配位置规划
- 降低不透明块

### 3.3 NPU代码生成

**MLIR → NPU IR**：通过tladapter passes转换

**NPU IR → 二进制**：使用bishengir-compile编译器

关键编译选项：

- `--enable-auto-multi-buffer=true`：自动多缓冲区优化
- `--enable-triton-kernel-compile=true`：Triton风格内核编译
- `--enable-hivm-compile=true`：HIVM编译流程

### 3.4 包装器生成

生成C++包装器代码，包含：

- 参数结构定义
- 内核启动函数
- Python调用入口

***

## 4. 运行时执行流程

```
Python调用 kernel(a, b, c)
    ↓
JitKernel_NPU.__call__()
    ↓
计算网格维度 (_calcu_grid)
    ↓
构建参数列表
    ↓
NPUUtils.load_binary()
    ↓
rtDevBinaryRegister() + rtFunctionRegister()
    ↓
rtKernelLaunch()
    ↓
NPU硬件执行
```

**关键运行时API**：

- `rtSetDevice()`：设置设备
- `rtDevBinaryRegister()`：注册二进制
- `rtFunctionRegister()`：注册函数
- `rtKernelLaunch()`：启动内核

***

## 5. 开发者快速上手

### 5.1 环境配置

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
export TILELANG_DUMP_IR=1  # 调试用
```

### 5.2 编写NPU内核示例

```python
import torch
import tilelang
import tilelang.language as T

N = 1024

@tilelang.jit(target="npuir")
def vec_add(N, block_N):
    n_num = N // block_N
    
    @T.prim_func
    def main(
        A: T.Tensor((N), "float32"),
        B: T.Tensor((N), "float32"),
        C: T.Tensor((N), "float32")
    ):
        with T.Kernel(n_num, is_npu=True) as (cid, _):
            A_VEC = T.alloc_ub((block_N), "float32")
            B_VEC = T.alloc_ub((block_N), "float32")
            C_VEC = T.alloc_ub((block_N), "float32")
            
            T.copy(A[cid * block_N], A_VEC)
            T.copy(B[cid * block_N], B_VEC)
            T.vadd(A_VEC, B_VEC, C_VEC)
            T.copy(C_VEC, C[cid * block_N])
    
    return main

# 使用
func = vec_add(N, 256)
a = torch.randn(N).float().npu()
b = torch.randn(N).float().npu()
c = torch.zeros(N).float().npu()
func(a, b, c)
```

### 5.3 性能优化要点

1. **分块大小**：根据L1/L0C缓冲区大小选择
2. **内存布局**：使用`load_nd2nz`进行ND到NZ格式转换
3. **流水线**：利用双缓冲隐藏内存延迟
4. **调试**：使用`TILELANG_DUMP_IR=1`和`T.print()`

### 5.4 常见问题

| 问题    | 解决方案                                     |
| ----- | ---------------------------------------- |
| 编译失败  | 检查ASCEND\_HOME\_PATH和bishengir-compile路径 |
| 运行时错误 | 验证输入张量的设备和数据类型                           |
| 性能问题  | 分析内存访问模式，优化分块策略                          |

***

## 6. 关键技术点

### 动态形状处理

```python
# 使用 T.symbolic() 定义符号变量
# 编译时提升为PrimFunc参数，运行时从输入张量形状推断值
@tilelang.jit(target="npuir")
def dynamic_matmul(block_M, block_N, K_L1):
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")
    
    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),  # M, K是符号变量
        B: T.Tensor((K, N), "float16"),  # K, N是符号变量
        C: T.Tensor((M, N), "float16")   # M, N是符号变量
    ):
        # 内核实现
        ...
    return main

# 使用：M, N, K在运行时从输入张量形状推断
func = dynamic_matmul(128, 256, 16)
a = torch.randn(1024, 2048).half().npu()  # M=1024, K=2048
b = torch.randn(2048, 512).half().npu()   # K=2048, N=512
c = torch.randn(1024, 512).half().npu()   # M=1024, N=512
func(a, b, c)
```

### 网格维度计算

根据符号变量和输入张量形状，在运行时计算并行执行的block数量。

### 工作区管理

NPU内核执行时的临时存储空间，从编译后的内核中自动获取大小。

### 内核缓存机制

TileLang使用多级缓存机制避免重复编译：

**缓存层级**：
1. **内存缓存**：运行时内存中的字典缓存，访问最快
2. **磁盘缓存**：持久化到磁盘，跨会话复用

**缓存键生成**：
```python
key_data = {
    "version": __version__,
    "func": sha256(func_binary).hexdigest(),
    "out_idx": tuple(out_idx),
    "target": str(target),
    "execution_backend": execution_backend,
    "pass_configs": pass_configs,
}
key = sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
```

**缓存文件结构**：
```
~/.tilelang/cache/
  ├── <hash_key>/
  │   ├── kernel.mlir          # MLIR中间表示
  │   ├── wrapped_kernel.o     # 编译后的目标文件
  │   ├── kernel_lib.so        # 共享库
  │   ├── params.pkl           # 参数信息
  │   ├── npu_utils.so         # NPU工具库
  │   ├── main.so              # 启动器
  │   └── metadata.pkl         # 元数据
```

**缓存控制环境变量**：
| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `TILELANG_CACHE_DIR` | `~/.tilelang/cache` | 缓存目录路径 |
| `TILELANG_DISABLE_CACHE` | `0` | 禁用缓存（调试用） |
| `TILELANG_CLEAR_CACHE` | `0` | 启动时清空缓存 |

**缓存API**：
```python
import tilelang.cache

# 获取缓存目录
cache_dir = tilelang.cache.get_cache_dir()

# 设置缓存目录
tilelang.cache.set_cache_dir("/path/to/cache")

# 清空缓存
tilelang.cache.clear_cache()
```

**缓存流程**：
```
编译请求
  ↓
生成缓存键
  ↓
检查内存缓存 → 命中 → 返回
  ↓ 未命中
检查磁盘缓存 → 命中 → 加载到内存 → 返回
  ↓ 未命中
执行编译
  ↓
保存到磁盘缓存
  ↓
保存到内存缓存
  ↓
返回结果
```

***

## 附录：编译Pass流程

```
PrimFunc
  → BindTarget
  → Simplify
  → RemoveNoOp
  → NpuLoopVectorize
  → PlanAndUpdateBufferAllocationLocation
  → LowerOpaqueBlock
  → RemoveNoOp
  → MLIR (NPU IR)
```

***

本文档保留了TileLang-Ascend JIT编译的核心概念、架构和关键流程，适合开发者快速理解和上手。详细实现请参考完整版文档和源码。
