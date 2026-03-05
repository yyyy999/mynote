# TileRT 调研报告

## 一、TileRT 是什么

### 1.1 定义
TileRT（Tile Runtime）是一个专为大型语言模型（LLM）设计的**超低延迟推理运行时系统**。其核心目标是实现毫秒级的每输出令牌时间（TPOT），通过细粒度的瓦片级任务分解和优化，在高端GPU上实现极高的解码速度。

### 1.2 核心特性
- **超低延迟优先**：专注于降低单个token的生成延迟
- **瓦片级运行时引擎**：将LLM算子分解为细粒度瓦片级任务
- **Multi-Token Prediction (MTP)**：支持推测解码技术，提升有效吞吐量
- **FP8量化支持**：支持FP8精度计算，降低显存占用和计算开销

### 1.3 性能表现
- **GLM-5-FP8**：在8×NVIDIA B200 GPU上达到 **500 tokens/s**
- **DeepSeek-V3.2**：在8×NVIDIA B200 GPU上达到 **600 tokens/s**

### 1.4 支持的模型
- GLM-5
- DeepSeek-V3.2

### 1.5 适用场景
- 高频交易系统
- 实时对话系统
- 低延迟推理服务

---

## 二、TileRT 的核心架构

### 2.1 整体架构

```
TileRT 架构层次
├── Python 层（应用层）
│   ├── models/              # 模型定义
│   │   ├── base.py          # 基础模块类
│   │   ├── deepseek_v3_2/   # DeepSeek模型实现
│   │   └── glm_5/           # GLM-5模型实现
│   ├── generator.py         # 文本生成器
│   ├── benchmark/           # 性能基准测试
│   └── profiler/            # 性能分析工具
├── C++ 层（运行时层）
│   └── libtilert.so         # 核心运行时库
│       ├── 调度器
│       ├── 内存管理器
│       ├── 缓存管理器
│       └── CUDA内核
└── TileLang 层（算子层）
    └── tilelang.jit         # 核心算子编译
        ├── act_quant_kernel
        ├── fp8_gemm_kernel
        └── 其他优化内核
```

### 2.2 核心组件

#### 2.2.1 Python层核心模块

**TileRTModule基类** (`python/models/base.py`)
- **目的**：定义所有TileRT模块的统一接口
- **核心功能**：
  - 权重管理（初始化、转换、加载）
  - 前向传播接口（golden_forward、tilert_forward）
  - 序列化支持
  - 性能分析集成

**生成器** (`python/models/deepseek_v3_2/generator.py`)
- **目的**：实现端到端的文本生成流程
- **核心功能**：
  - 支持标准生成和MTP生成
  - 支持prefill-decode分离
  - 支持缓存注入（inject_cache）
  - 支持多种采样策略（top-p、top-k）

**模型参数** (`python/models/deepseek_v3_2/model_args.py`)
- **目的**：定义模型超参数和配置
- **核心参数**：
  - 架构参数：dim=7168, n_layers=61, n_heads=128
  - MoE参数：n_routed_experts=256, n_activated_experts=8
  - MLA参数：kv_lora_rank=512, qk_nope_head_dim=128
  - 量化参数：block_size=128

#### 2.2.2 核心算子实现

**FP8 GEMM内核** (`python/models/deepseek_v3_2/refs/kernel.py`)
```python
@tilelang.jit(
    out_idx=[-1],
    target="cuda",
    num_stages=STAGES,
    threads=THREADS,
)
def fp8_gemm_kernel(
    A: torch.Tensor,  # FP8激活
    B: torch.Tensor,  # FP8权重
    C: torch.Tensor,  # BF16输出
    Ascale: torch.Tensor,
    Bscale: torch.Tensor,
):
    # 实现FP8矩阵乘法
    # 支持权重反量化
    # 优化内存访问模式
```

**激活量化内核**
```python
@tilelang.jit(...)
def act_quant_kernel(
    X: torch.Tensor,      # BF16输入
    XQ: torch.Tensor,     # FP8输出
    XScale: torch.Tensor, # 量化scale
):
    # 实现激活量化
    # 支持block-wise量化
```

#### 2.2.3 运行时引擎（C++层）

**核心功能**：
- **调度器**：管理多流、多worker的任务调度
- **内存管理器**：优化显存分配和复用
- **缓存管理器**：管理KV缓存、PE缓存、KI缓存
- **CUDA内核**：高度优化的CUDA实现

**调度策略**：
```
Worker定义：
- Init: 初始化任务
- Prefetch: 预取任务
- Compute: 计算任务
- ExtraTask1/SyncIo: 同步IO
- ExtraTask2/IoP0: IO端口0
- ExtraTask3/IoP2: IO端口2
- ExtraTask4: 额外任务4
- ExtraTask5: 额外任务5
```

---

## 三、TileRT 如何使用

### 3.1 安装步骤

#### 3.1.1 Docker环境搭建
```bash
# 1. 构建Docker镜像
docker build -t tilert:latest .

# 2. 启动容器
docker run --gpus all -it --rm \
    -v /path/to/weights:/weights \
    -v /path/to/output:/output \
    tilert:latest
```

#### 3.1.2 权重转换
```bash
# 转换DeepSeek-V3.2权重
python -m tilert.models.preprocess.weight_converter \
    --model_path /path/to/deepseek_weights \
    --output_path /output/deepseek_v3_2_tilert \
    --model_type deepseek_v3_2

# 转换GLM-5权重
python -m tilert.models.preprocess.weight_converter \
    --model_path /path/to/glm5_weights \
    --output_path /output/glm5_tilert \
    --model_type glm_5
```

### 3.2 基本使用

#### 3.2.1 标准生成（无MTP）
```python
from tilert.models.deepseek_v3_2.generator import DSAv32Generator
from tilert.models.deepseek_v3_2.model_args import ModelArgs

# 初始化
model_args = ModelArgs()
generator = DSAv32Generator(
    model_args=model_args,
    max_new_tokens=100,
    model_weights_dir="/path/to/weights",
    with_mtp=False
)

# 加载权重
generator.init()
generator.from_pretrained()

# 生成文本
prompt = "你好，请介绍一下TileRT"
result, time_list, _ = generator.generate(prompt, print_log=True)

# 清理
generator.cleanup()
```

#### 3.2.2 MTP生成（推测解码）
```python
# 初始化（启用MTP）
generator = DSAv32Generator(
    model_args=model_args,
    max_new_tokens=100,
    model_weights_dir="/path/to/weights",
    with_mtp=True  # 启用MTP
)

# 生成文本
result, time_list, accepted_counts = generator.generate(
    prompt,
    print_log=True,
    with_mtp=True
)

# 性能统计
print(f"平均接受token数: {sum(accepted_counts)/len(accepted_counts):.2f}")
```

#### 3.2.3 Prefill-Decode分离
```python
# 注入外部预填充缓存
layer_caches = []
for layer_id in range(61):
    ki = load_ki_for_layer(layer_id)  # [seqlen, 128]
    kv = load_kv_for_layer(layer_id)  # [seqlen, 512]
    pe = load_pe_for_layer(layer_id)  # [seqlen, 64]
    layer_caches.append((ki, kv, pe))

# 注入缓存
generator.inject_cache(layer_caches, start_pos=0)

# 设置当前位置
generator.set_cur_pos(seqlen)

# 继续生成
result, time_list, _ = generator.generate("", print_log=True)
```

### 3.3 高级功能

#### 3.3.1 采样参数调整
```python
generator.update_sampling_params(
    temperature=0.8,
    top_p=0.95,
    top_k=256,
    use_topp=True
)
```

#### 3.3.2 性能分析
```python
from tilert.profiler.utils import parse_profile_log_tensor

# 启用性能分析
generator.decode_layer.flag_enable_profiling_log = True

# 生成后分析
parse_profile_log_tensor(
    profile_logs_tensor,
    out_path="/output/profile.xlsx",
    inst2opname=op_list
)
```

---

## 四、TileRT 与 TileLang 的配合方式

### 4.1 TileLang 的角色

TileLang 是一个**高性能算子编译框架**，TileRT 在其基础上构建了完整的运行时系统。

#### 4.1.1 TileLang 提供的能力
- **JIT编译**：`tilelang.jit` 装饰器，将Python函数编译为高效CUDA内核
- **内存布局优化**：自动优化张量内存访问模式
- **并行化**：自动并行化计算任务
- **流水线**：支持多阶段流水线执行

#### 4.1.2 TileRT 的扩展
- **运行时管理**：调度、内存、缓存管理
- **端到端流程**：从输入到输出的完整推理流程
- **模型抽象**：统一的模型接口和权重管理
- **高级特性**：MTP、prefill-decode分离等

### 4.2 核心算子集成

#### 4.2.1 FP8矩阵乘法
```python
# TileRT使用TileLang实现FP8 GEMM
@tilelang.jit(
    out_idx=[-1],
    target="cuda",
    num_stages=2,
    threads=128,
)
def fp8_gemm_kernel(A, B, C, Ascale, Bscale):
    # 1. 加载FP8权重
    # 2. 反量化为BF16
    # 3. 执行矩阵乘法
    # 4. 量化输出
```

**调用链**：
```
generator.generate()
  -> ShowHandsDSALayer.forward()
    -> fp8_gemm_kernel (TileLang编译)
      -> CUDA内核执行
```

#### 4.2.2 激活量化
```python
@tilelang.jit(...)
def act_quant_kernel(X, XQ, XScale):
    # 1. 计算block-wise最大值
    # 2. 量化为FP8
    # 3. 存储scale
```

### 4.3 编译参数优化

TileRT 针对不同算子设置了不同的编译参数：

```python
# FP8 GEMM: 2阶段流水线，128线程
@tilelang.jit(num_stages=2, threads=128)

# 激活量化: 1阶段，64线程
@tilelang.jit(num_stages=1, threads=64)

# 索引内核: 禁用TMA优化
@tilelang.jit(enable_tma=False)
```

### 4.4 权重转换与布局

TileRT 实现了复杂的权重转换逻辑，以适配TileLang编译的内核：

```python
class RMSNormProjQAKVAKIWeightsConverter:
    @staticmethod
    def common_to_tilert_fp8(...):
        # 1. 权重重排（swizzle）
        # 2. scale布局转换
        # 3. 拼接权重和scale
        # 4. 转换为FP8格式
```

---

## 五、在 tilelang-ascend 项目上的应用方案

### 5.1 硬件差异分析

#### 5.1.1 NVIDIA vs Ascend
| 维度 | NVIDIA | Ascend | 影响 |
|------|--------|--------|------|
| **架构** | CUDA | CANN | 需要重写内核 |
| **计算单元** | Tensor Core | Cube单元 | 矩阵乘法优化不同 |
| **内存模型** | CUDA内存模型 | Ascend内存模型 | 内存管理需适配 |
| **并行模型** | CUDA线程模型 | Ascend并行模型 | 并行策略需调整 |
| **量化支持** | FP8原生支持 | FP8支持需确认 | 量化策略可能需调整 |

#### 5.1.2 关键挑战
1. **内核移植**：TileLang编译的CUDA内核无法直接在Ascend上运行
2. **内存管理**：Ascend的内存模型与CUDA不同
3. **调度策略**：Ascend的并行模型与CUDA不同
4. **性能优化**：需要针对Ascend架构重新优化

### 5.2 改造方案

#### 5.2.1 整体策略
```
TileRT-Ascend 架构
├── Python层（保留）
│   ├── models/              # 模型定义（保留）
│   ├── generator.py         # 生成器（保留）
│   └── 权重转换（需适配）
├── C++层（重写）
│   └── libtilert_ascend.so  # Ascend运行时库
│       ├── Ascend调度器
│       ├── Ascend内存管理器
│       └── Ascend缓存管理器
└── TileLang层（替换）
    └── tilelang_ascend.jit  # Ascend算子编译
        ├── ascend_gemm_kernel
        ├── ascend_quant_kernel
        └── 其他Ascend优化内核
```

#### 5.2.2 分阶段实施

**阶段一：基础算子移植（优先级：高）**
```python
# 1. FP8矩阵乘法
@tilelang_ascend.jit(target="ascend")
def ascend_fp8_gemm_kernel(A, B, C, Ascale, Bscale):
    # 使用Ascend Cube单元
    # 实现FP8矩阵乘法

# 2. 激活量化
@tilelang_ascend.jit(target="ascend")
def ascend_act_quant_kernel(X, XQ, XScale):
    # 实现block-wise量化

# 3. RMSNorm
@tilelang_ascend.jit(target="ascend")
def ascend_rmsnorm_kernel(X, Gamma, Y):
    # 实现RMSNorm
```

**阶段二：运行时系统适配（优先级：高）**
```cpp
// 1. Ascend调度器
class AscendScheduler {
    // 管理Ascend流、事件
    // 实现任务调度
};

// 2. Ascend内存管理器
class AscendMemoryManager {
    // 管理Ascend内存池
    // 实现内存复用
};

// 3. Ascend缓存管理器
class AscendCacheManager {
    // 管理KV缓存
    // 优化缓存访问
};
```

**阶段三：权重转换适配（优先级：中）**
```python
# 适配权重转换逻辑
class AscendWeightsConverter:
    @staticmethod
    def common_to_ascend_fp8(...):
        # 1. 权重重排（适配Ascend内存布局）
        # 2. scale布局转换
        # 3. 拼接权重和scale
        # 4. 转换为Ascend支持的格式
```

**阶段四：高级特性支持（优先级：低）**
```python
# 1. MTP支持
class AscendMTPGenerator:
    # 实现MTP推测解码

# 2. Prefill-Decode分离
class AscendPrefillDecode:
    # 实现prefill-decode分离
```

### 5.3 关键技术点

#### 5.3.1 算子实现
```python
# FP8矩阵乘法（Ascend版本）
def ascend_fp8_gemm(A, B, C, Ascale, Bscale):
    """
    输入：
        A: [M, K] FP8激活
        B: [N, K] FP8权重
        C: [M, N] BF16输出
        Ascale: [M, K//block_size] scale
        Bscale: [N, K//block_size] scale
    
    实现：
        1. 使用Ascend Cube单元执行矩阵乘法
        2. 支持FP8输入、BF16输出
        3. 优化内存访问模式
    """
    # 使用Ascend API
    aclnn_fp8_gemm(A, B, C, Ascale, Bscale)
```

#### 5.3.2 内存管理
```cpp
// Ascend内存池
class AscendMemoryPool {
public:
    // 分配内存
    void* allocate(size_t size);
    
    // 释放内存
    void deallocate(void* ptr);
    
    // 内存复用
    void reuse(void* ptr, size_t new_size);
    
private:
    std::vector<MemoryBlock> blocks_;
    std::unordered_map<void*, MemoryBlock> ptr_to_block_;
};
```

#### 5.3.3 调度策略
```cpp
// Ascend任务调度
class AscendTaskScheduler {
public:
    // 提交任务
    void submit_task(Task task);
    
    // 执行任务
    void execute();
    
private:
    // Ascend流管理
    std::vector<aclrtStream> streams_;
    
    // 任务队列
    std::queue<Task> task_queue_;
    
    // 事件管理
    std::vector<aclrtEvent> events_;
};
```

### 5.4 性能优化建议

#### 5.4.1 内存访问优化
- **连续内存访问**：确保张量内存布局连续
- **内存对齐**：对齐到Ascend内存边界（如64字节）
- **缓存友好**：优化数据访问模式，提高缓存命中率

#### 5.4.2 并行优化
- **多流并行**：利用Ascend多流能力
- **任务流水线**：实现计算与IO重叠
- **负载均衡**：均衡分配计算任务

#### 5.4.3 计算优化
- **Cube单元利用**：充分利用Ascend Cube单元
- **量化优化**：优化FP8量化/反量化性能
- **内核融合**：融合多个小内核减少开销

### 5.5 实施路线图

#### 第一阶段（1-2个月）：基础功能
- [ ] 移植核心算子（FP8 GEMM、激活量化、RMSNorm）
- [ ] 实现基础运行时系统（调度、内存管理）
- [ ] 实现标准生成流程（无MTP）
- [ ] 验证正确性和基础性能

#### 第二阶段（2-3个月）：性能优化
- [ ] 优化算子性能
- [ ] 优化内存管理
- [ ] 优化调度策略
- [ ] 性能对标（目标：达到TileRT 70%以上性能）

#### 第三阶段（3-4个月）：高级特性
- [ ] 实现MTP支持
- [ ] 实现prefill-decode分离
- [ ] 支持更多模型
- [ ] 完善文档和测试

### 5.6 风险与挑战

#### 5.6.1 技术风险
1. **FP8支持**：Ascend对FP8的支持程度需确认
2. **性能差距**：Ascend与NVIDIA的性能差距
3. **工具链成熟度**：Ascend工具链的成熟度

#### 5.6.2 应对策略
1. **调研先行**：深入调研Ascend FP8支持
2. **渐进式开发**：先实现基础功能，再优化性能
3. **社区支持**：积极寻求Ascend社区支持

---

## 六、总结

### 6.1 TileRT 的核心价值
1. **超低延迟**：专注于降低单个token生成延迟
2. **高性能**：在高端GPU上实现500-600 tokens/s
3. **模块化设计**：清晰的架构分层，易于扩展
4. **TileLang集成**：充分利用TileLang的编译优化能力

### 6.2 在 tilelang-ascend 上的应用价值
1. **技术复用**：Python层架构可直接复用
2. **性能参考**：提供明确的性能目标和优化方向
3. **架构参考**：提供完整的运行时系统设计参考
4. **加速开发**：减少从零开始的设计和实现工作

### 6.3 关键成功因素
1. **算子性能**：核心算子的性能是关键
2. **内存管理**：高效的内存管理是基础
3. **调度优化**：合理的调度策略是保障
4. **工具链支持**：成熟的工具链是前提

---

## 附录

### A. TileRT 核心文件清单
```
TileRT/
├── python/
│   ├── __init__.py                    # 包初始化，加载libtilert.so
│   ├── tilert_init.py                 # TileRT初始化
│   ├── models/
│   │   ├── base.py                    # 基础模块类
│   │   ├── deepseek_v3_2/
│   │   │   ├── generator.py           # DeepSeek生成器
│   │   │   ├── model_args.py          # 模型参数
│   │   │   ├── modules/
│   │   │   │   ├── dsa.py             # DSA模块
│   │   │   │   ├── mlp.py             # MLP模块
│   │   │   │   └── end2end.py         # 端到端模块
│   │   │   ├── ops/
│   │   │   │   ├── flash_sparse_mla.py # Flash Sparse MLA
│   │   │   │   ├── rmsnorm_projx_wqkvia.py # RMSNorm投影
│   │   │   │   └── qkv_rope.py        # QKV RoPE
│   │   │   └── refs/
│   │   │       └── kernel.py          # TileLang内核
│   │   └── glm_5/                     # GLM-5实现
│   ├── profiler/
│   │   └── utils.py                   # 性能分析工具
│   └── utils.py                       # 工具函数
└── README.md                          # 项目文档
```

### B. 性能基准
| 模型 | GPU | 精度 | 性能 (tokens/s) | 延迟 (ms/token) |
|------|-----|------|-----------------|-----------------|
| GLM-5 | 8×B200 | FP8 | 500 | 2.0 |
| DeepSeek-V3.2 | 8×B200 | FP8 | 600 | 1.67 |

### C. 参考资料
- TileRT GitHub: https://github.com/tile-ai/tileRT
- TileLang文档: https://tilelang.readthedocs.io
- DeepSeek-V3论文: https://arxiv.org/abs/2412.19437
- Ascend CANN文档: https://www.hiascend.com/document