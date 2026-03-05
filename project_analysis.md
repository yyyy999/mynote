# TileScale 仓库结构分析文档

## 一、项目概述

TileScale 是 TileLang 的分布式扩展，将 TileLang 的 tile-level 编程模型扩展到多GPU、多节点乃至分布式芯片架构范围。它是一个分布式原生的领域特定语言(DSL)和编译器栈，专为下一代分布式架构上的深度学习设计。

---

## 二、核心架构：分层分布式架构 (HDA)

### 2.1 HDA 概念
TileScale 引入了统一的虚拟设备架构 —— **Hierarchical Distributed Architecture (HDA)**，用于抽象分布式系统：

- **计算单元**: 线程 → Warp → SM → GPU → Node
- **内存**: L0(寄存器) → L1(共享内存) → L2 → 全局内存
- **网络**: NoC(片内) → NVLink(片间) → InfiniBand(节点间)

### 2.2 编程接口层次
```
┌─────────────────────────────────────────────────────────┐
│                    Device Scale                          │
│  (多GPU/多节点级别的计算、内存、通信原语)                    │
├─────────────────────────────────────────────────────────┤
│                    Block Scale                           │
│  (SM级别的计算、内存、通信原语)                             │
├─────────────────────────────────────────────────────────┤
│                    Warp/Warpgroup Scale                  │
│  (Warp级别的计算、内存、通信原语)                           │
├─────────────────────────────────────────────────────────┤
│                    Thread Scale                          │
│  (线程级别的计算、内存原语)                                 │
└─────────────────────────────────────────────────────────┘
```

---

## 三、TileScale 在 TileLang 上的扩展实现

### 3.1 tilescale_ext 模块 (C++/Python 绑定)

**位置**: `tilescale_ext/` 和 `tilelang/utils/ts_ext/`

**功能**: 提供底层 CUDA/PyTorch 扩展，支持分布式张量操作

**核心函数**:
| 函数 | 功能 | 调用链 |
|------|------|--------|
| `tensor_from_ptr` | 从指针创建张量 | Python → tilescale_ext._C |
| `_create_tensor` | 创建分布式张量 | Python → tilescale_ext._C → CUDA malloc |
| `_create_ipc_handle` | 创建IPC句柄 | Python → tilescale_ext._C → CUDA IPC |
| `_sync_ipc_handles` | 同步IPC句柄 | Python → tilescale_ext._C → 多GPU同步 |
| `create_host_device_tensor` | 创建主机设备张量 | Python → tilescale_ext._C |

**源文件**:
- `tilelang/utils/ts_ext/ts_ext_bindings.cpp` - PyTorch C++ 绑定
- `tilelang/utils/ts_ext/tensor.cpp` - 张量操作实现
- `tilelang/utils/ts_ext/ipc_ops.cpp` - IPC 操作实现

### 3.2 分布式通信原语 (Distributed Primitives)

**位置**: `tilelang/language/distributed/` 和 `src/op/distributed.cc`

#### 3.2.1 核心通信操作

**Python 接口** (`tilelang/language/distributed/common.py`):
```python
# Warp级别通信
put_warp(src, dst, size, dst_pe)  # Warp级Put操作
get_warp(src, dst, size, src_pe)  # Warp级Get操作

# Block级别通信
put_block(src, dst, size, dst_pe)  # Block级Put操作
get_block(src, dst, size, src_pe)  # Block级Get操作

# 同步原语
wait_eq(value, expected, peer)  # 等待 value == expected
wait_ne(value, expected, peer)  # 等待 value != expected
```

**NVSHMEM 接口** (`tilelang/language/distributed/multi_device/nvshmem.py`):
```python
# PE(处理单元)管理
get_pe()        # 获取当前PE ID
get_pe_num()    # 获取PE总数

# 屏障同步
barrier_all()           # 全局屏障
barrier_all_block()     # Block级屏障
barrier_all_warp()      # Warp级屏障

# 内存操作
getmem_nbi_block()      # 非阻塞Block级Get
putmem_nbi_block()      # 非阻塞Block级Put
putmem_signal_nbi_block()  # 带信号的Put
```

#### 3.2.2 C++ IR 定义

**位置**: `src/op/distributed.h` 和 `src/op/distributed.cc`

定义了所有分布式操作的 TIR intrinsic:
```
GetPE / GetPENum / IntPE          # PE信息获取
BarrierAll / BarrierAllBlock / BarrierAllWarp  # 屏障同步
SyncAll / SyncAllBlock / SyncAllWarp          # 同步
Getmem / Putmem / GetmemNbi / PutmemNbi       # 内存传输
PutmemSignal / PutmemSignalNbi                # 带信号传输
Broadcast / Fcollect                          # 集合通信
```

### 3.3 远程拷贝操作 (Remote Copy)

**位置**: `src/op/remote_copy.h` 和 `src/op/remote_copy.cc`

**核心类**:
- `PutOpNode`: Put操作节点 (本地 → 远程)
- `GetOpNode`: Get操作节点 (远程 → 本地)

**关键属性**:
```cpp
class PutOpNode : public TileOperatorNode {
  PrimExpr src_addr;           // 源地址
  PrimExpr dst_addr;           // 目标地址
  PrimExpr copy_size;          // 拷贝大小
  PrimExpr dst_pe;             // 目标PE
  int unroll_factor;           // 展开因子
  std::string scope;           // 作用域: {warp, block}
  bool enable_aggressive_vectorize;  // 激进向量化
};
```

### 3.4 同步原语 (Synchronization)

**位置**: `src/op/sync.h` 和 `src/op/sync.cc`

**核心类**:
- `WaitOpNode`: 等待操作节点
- `BarrierBlocksOpNode`: Block级屏障操作节点

**功能**:
- 条件等待: `wait_eq`, `wait_ne`, `wait_ge`, `wait_le`, `wait_gt`, `wait_lt`
- 系统级屏障: 支持跨GPU的同步

### 3.5 分布式运行时 (Distributed Runtime)

**位置**: `src/runtime/tilescale_cuda_module.h` 和 `src/runtime/tilescale_cuda_module.cc`

**功能**: 扩展 TVM 的 CUDA 模块，支持分布式表初始化

```cpp
// 创建 TileScale 扩展的 CUDA 模块
ffi::Module TileScaleCUDAModuleCreate(
    std::string data, 
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string cuda_source
);
```

**特殊功能**:
- `__tilescale_init_distributed_table`: 初始化分布式表，将主机数据拷贝到设备的 meta_data 符号

### 3.6 分布式工具函数

**位置**: `tilelang/distributed/utils.py`

**核心函数**:
| 函数 | 功能 |
|------|------|
| `init_dist` | 初始化分布式环境 |
| `init_distributed` | 初始化分布式+NVSHMEM |
| `create_tensor` | 创建分布式张量 |
| `get_local_ipc_handle` | 获取本地IPC句柄 |
| `create_dist_tensor` | 创建分布式张量(跨GPU) |

### 3.7 pynvshmem 封装

**位置**: `tilelang/distributed/pynvshmem/`

**功能**: 提供 NVSHMEM 的 Python 封装，支持:
- NVSHMEM 初始化
- 对称堆内存分配
- 设备端通信 API

---

## 四、TileScale 扩展功能总结

### 4.1 主要功能

1. **分层通信原语**
   - 支持 Warp/Block/Device 级别的 Put/Get 操作
   - 支持阻塞和非阻塞两种模式

2. **分布式同步**
   - 全局屏障 (barrier_all)
   - 条件等待 (wait_eq/ne/ge/le/gt/lt)
   - 内存栅栏 (fence/quiet)

3. **IPC 通信**
   - CUDA IPC 句柄创建和同步
   - 跨 GPU 内存访问

4. **NVSHMEM 集成**
   - 对称堆内存管理
   - 高性能远程内存访问

5. **集合通信**
   - AllGather
   - All2All
   - Broadcast
   - Fcollect

### 4.2 编程模型扩展

TileScale 在 TileLang 基础上引入了:

1. **T.Scale 原语** (规划中，见 README)
   ```python
   with T.Scale("device") as dev_id, dev_num:
       # 设备级并行
   ```

2. **分布式内存视图**
   ```python
   A_global = T.view(A, layout=T.FullCol)  # 列分片
   B_global = T.view(B, layout=T.FullRow)  # 行分片
   C_global = T.view(C, layout=T.Replica)  # 副本
   ```

3. **层次化内存分配**
   ```python
   A_local = T.alloc((block_M, block_K), dtype, level="l0")  # 寄存器
   A_shared = T.alloc((block_M, block_K), dtype, level="l1") # 共享内存
   ```

---

## 五、目录结构

```
tilescale/
├── tilescale_ext/                    # Python 包入口
│   └── __init__.py                   # 导出分布式扩展函数
│
├── tilelang/
│   ├── language/
│   │   └── distributed/              # 分布式语言接口
│   │       ├── common.py             # 通用分布式原语
│   │       └── multi_device/
│   │           └── nvshmem.py        # NVSHMEM 接口
│   │
│   ├── distributed/                  # 分布式运行时支持
│   │   ├── utils.py                  # 分布式工具函数
│   │   └── pynvshmem/                # NVSHMEM Python 封装
│   │
│   └── utils/
│       └── ts_ext/                   # tilescale_ext 源码
│           ├── ts_ext_bindings.cpp   # PyTorch 绑定
│           ├── tensor.cpp            # 张量操作
│           └── ipc_ops.cpp           # IPC 操作
│
├── src/
│   ├── op/
│   │   ├── distributed.h/cc          # 分布式 IR 定义
│   │   ├── remote_copy.h/cc          # 远程拷贝操作
│   │   └── sync.h/cc                 # 同步操作
│   │
│   └── runtime/
│       └── tilescale_cuda_module.h/cc # 分布式 CUDA 模块
│
└── examples/
    └── distributed/                  # 分布式示例
        ├── example_allgather.py      # AllGather 示例
        ├── example_all_to_all.py     # All2All 示例
        └── ...
```

---

## 六、调用链总结

### 6.1 分布式张量创建
```
Python: create_tensor(shape, dtype)
    ↓
tilescale_ext._C._create_tensor
    ↓
C++: tensor.cpp (CUDA malloc)
    ↓
返回 PyTorch Tensor
```

### 6.2 跨 GPU 通信
```
Python: T.putmem_nbi_block(dst, src, size, pe)
    ↓
TIR: tir.call_intrin("tl.PutmemNbiBlock", ...)
    ↓
C++ IR: src/op/distributed.cc (Op注册)
    ↓
Codegen: src/target/codegen_cuda.cc
    ↓
CUDA: nvshmem_putmem_nbi_block()
```

### 6.3 IPC 同步
```
Python: _sync_ipc_handles(rank, device_ids, ...)
    ↓
tilescale_ext._C._sync_ipc_handles
    ↓
C++: ipc_ops.cpp (cudaIpcOpenMemHandle)
    ↓
GPU间内存映射完成
```

---

## 七、与 TileLang 的关系

TileScale 是 TileLang 的**超集扩展**:

1. **继承**: 完整继承 TileLang 的所有 tile-level 编程能力
2. **扩展**: 新增分布式通信、同步、IPC 等能力
3. **统一**: 使用相同的 TIR IR 和编译流程
4. **兼容**: TileLang 程序可直接在 TileScale 上运行

---

## 八、分布式示例分析

### 8.1 示例概览

| 示例 | 功能 | 通信模式 | 应用场景 |
|------|------|----------|----------|
| `example_allgather.py` | AllGather 集合通信 | Ring AllGather | 张量并行 |
| `example_all_to_all.py` | All2All 集合通信 | Signal同步 | MoE专家路由 |
| `example_allgather_gemm.py` | AllGather + GEMM融合 | 通信计算重叠 | 分布式矩阵乘 |
| `example_summa.py` | SUMMA分布式矩阵乘 | 2D Mesh广播 | 大规模矩阵乘 |
| `example_cannon.py` | Cannon分布式矩阵乘 | 环形移位 | 大规模矩阵乘 |
| `example_simple_shift.py` | 简单环形移位 | Put操作 | 基础通信测试 |
| `primitives/example_put_block.py` | Block级Put原语 | 点对点发送 | 基础原语测试 |
| `primitives/example_get_block.py` | Block级Get原语 | 点对点接收 | 基础原语测试 |
| `primitives/example_sync.py` | IPC同步原语 | 内存映射 | 基础原语测试 |
| `example_nvshmem.py` | NVSHMEM基础测试 | PE查询 | 环境验证 |

---

### 8.2 核心示例详解

#### 8.2.1 AllGather 示例

**文件**: [example_allgather.py](file:///d:/work/tilescale/examples/distributed/example_allgather.py)

**功能**: 实现 Ring AllGather 算法，每个 GPU 将本地数据发送给其他所有 GPU

**核心代码**:
```python
@T.prim_func
def a2a_split(A: T.Tensor((M_per_rank, N), dtype), B: T.Tensor((M, N), dtype)):
    with T.Kernel(M_per_rank // block_M, PE_num - 1, threads=threads) as (bx, by):
        mype = T.get_pe()      # 获取当前PE ID
        npes = T.get_pe_num()  # 获取PE总数
        
        A_shared = T.alloc_shared((block_M, N), dtype)
        local_base = bx * block_M
        global_base = M_per_rank * mype + local_base
        T.copy(A[local_base : local_base + block_M, :], A_shared)
        
        # Ring AllGather: 发送给 (mype + by + 1) % npes
        peer = (mype + by + 1) % npes
        T.putmem_nbi_block(
            T.address_of(B[global_base, 0]), 
            T.address_of(A_shared[0, 0]), 
            block_M * N * dtype_map[dtype].itemsize, 
            peer
        )
```

**调用链**:
```
Python: T.putmem_nbi_block()
    ↓
TIR intrinsic: tl.PutmemNbiBlock
    ↓
Codegen → CUDA: nvshmem_putmem_nbi_block()
    ↓
NVSHMEM: RDMA写入远程GPU内存
```

**特点**:
- 使用 NVSHMEM 对称堆内存 (`pynvshmem.nvshmem_create_tensor`)
- 非阻塞 Put 操作实现流水线
- Ring 算法减少通信开销

---

#### 8.2.2 All2All 示例 (MoE场景)

**文件**: [example_all_to_all.py](file:///d:/work/tilescale/examples/distributed/example_all_to_all.py)

**功能**: 实现 MoE (Mixture of Experts) 场景下的 All2All 通信，支持变长数据传输

**核心代码**:
```python
@T.prim_func
def main(
    data_src: T.Tensor((TOKEN_NUM * TOPK, HIDDEN), "float16"),
    signal: T.Tensor((PE_num,), "uint64"),
    splits_cumsum: T.Tensor((EXPERT_NUM + 1,), "int32"),
    data_dst: T.Tensor((TOKEN_NUM * TOPK, HIDDEN), "float16"),
):
    with T.Kernel(PE_num, threads=128) as (bx):
        peer = bx
        mype[0] = T.get_pe()
        m_start[0] = splits_cumsum[peer * EXPERTS_PER_RANK]
        m_end[0] = splits_cumsum[(peer + 1) * EXPERTS_PER_RANK]

        # 发送数据到目标PE
        T.putmem_nbi_block(
            T.address_of(data_dst[0, 0]), 
            T.address_of(data_src[m_start[0], 0]), 
            (m_end[0] - m_start[0]) * HIDDEN * 2, 
            peer
        )

        T.fence()  # 内存栅栏

        # Signal同步机制
        if tx == 0:
            T.signal_op(T.address_of(signal[mype[0]]), 99, 9, peer)
            T.signal_wait_until(T.address_of(signal[peer]), 0, 99)
```

**特点**:
- 支持变长数据传输 (基于 `splits_cumsum`)
- Signal 机制实现同步
- 适用于 MoE 专家路由场景

---

#### 8.2.3 AllGather + GEMM 融合

**文件**: [example_allgather_gemm.py](file:///d:/work/tilescale/examples/distributed/example_allgather_gemm.py)

**功能**: 将 AllGather 通信与 GEMM 计算融合，实现通信计算重叠

**核心代码**:
```python
@T.prim_func
def main(A, A_ag, B, signal, C):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        # 1. 先将本地数据拷贝到全局缓冲区
        T.copy(A[by * block_M, bx * block_K], A_shared)
        T.copy(A_shared, A_ag[mype[0] * M, bx * block_K])
        
        # 2. Ring广播到其他PE (带Signal)
        for k in T.serial(PE_num - 1):
            peer[0] = (mype[0] + 1 + k) % npes[0]
            T.putmem_signal_nbi_block(
                T.address_of(A_ag[mype[0] * M, 0]),
                T.address_of(A[0, 0]),
                block_M * block_K * 2,
                T.address_of(signal[k]),
                k + 1, 9, peer[0]
            )
        
        # 3. 等待Signal
        for k in T.serial(PE_num - 1):
            T.signal_wait_until(T.address_of(signal[k]), 0, k + 1)

        # 4. 执行GEMM
        for bk in T.serial(PE_num):
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A_ag[bk * M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bk * M, bx * block_N])
```

**特点**:
- 通信计算流水线化
- `putmem_signal_nbi_block` 实现带通知的发送
- 适用于张量并行矩阵乘

---

#### 8.2.4 SUMMA 分布式矩阵乘

**文件**: [example_summa.py](file:///d:/work/tilescale/examples/distributed/example_summa.py)

**功能**: 实现 SUMMA (Scalable Universal Matrix Multiplication Algorithm) 算法

**算法原理**:
```
2D Mesh: PE(pe_mn, pe_k)
- pe_mn = mype // MESH  (行/列索引)
- pe_k = mype % MESH   (K维度索引)

每个迭代 ko:
1. PE(pe_mn, ko) 广播 A 到同行PE
2. PE(pe_mn, ko) 广播 B 到同列PE
3. 所有PE执行本地GEMM
```

**核心代码**:
```python
for ko in T.serial(MESH):
    # 广播 A (当前持有A的PE发送给同行其他PE)
    if pe_k == ko:
        for peer_k in T.serial(MESH):
            T.putmem_signal_nbi_block(
                T.address_of(A[(ko + 1) % 2, ...]),
                T.address_of(A[ko % 2, ...]),
                ...,
                pe_mn * MESH + peer_k
            )

    # 广播 B (当前持有B的PE发送给同列其他PE)
    if pe_k == ko:
        for peer_k in T.serial(MESH):
            T.putmem_signal_nbi_block(
                T.address_of(B[(ko + 1) % 2, ...]),
                T.address_of(B[ko % 2, ...]),
                ...,
                pe_mn * MESH + peer_k
            )

    # 等待数据就绪
    T.signal_wait_until(...)

    # 执行本地GEMM
    for ki in T.Pipelined(T.ceildiv(K_local, block_K), num_stages=4):
        T.copy(A[ko % 2, ...], A_shared)
        T.copy(B[ko % 2, ...], B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

**特点**:
- 2D Mesh 拓扑
- 双缓冲 (A[0] 和 A[1]) 实现流水线
- 适用于大规模分布式矩阵乘

---

#### 8.2.5 Cannon 分布式矩阵乘

**文件**: [example_cannon.py](file:///d:/work/tilescale/examples/distributed/example_cannon.py)

**功能**: 实现 Cannon 算法，通过环形移位实现分布式矩阵乘

**算法原理**:
```
初始移位:
- A 矩阵: 每行向左移位 pe_k 次
- B 矩阵: 每列向上移位 pe_mn 次

每次迭代:
- 本地 GEMM
- A 向左移位一次
- B 向上移位一次
```

**核心代码**:
```python
# 计算移位目标PE
a_peer_from[0] = (mype[0] + 1) % MESH + MESH * (mype[0] // MESH)  # 右邻居
a_peer_to[0] = (mype[0] - 1 + MESH) % MESH + MESH * (mype[0] // MESH)  # 左邻居
b_peer_from[0] = (mype[0] + MESH) % npes[0]  # 下邻居
b_peer_to[0] = (mype[0] - MESH + npes[0]) % npes[0]  # 上邻居

for ko in T.serial(MESH):
    # 等待数据到达
    T.signal_wait_until(...)
    
    # 发送 A 到左邻居
    T.putmem_signal_nbi_block(..., a_peer_to[0])
    # 发送 B 到上邻居
    T.putmem_signal_nbi_block(..., b_peer_to[0])

    # 本地 GEMM
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

    # 通知数据已发送
    T.signal_op(..., a_peer_from[0])
    T.signal_op(..., b_peer_from[0])
```

**特点**:
- 环形移位模式
- 固定通信模式，适合硬件优化
- 与 SUMMA 相比，通信量更小

---

#### 8.2.6 基础原语示例

**Put Block 原语**: [primitives/example_put_block.py](file:///d:/work/tilescale/examples/distributed/primitives/example_put_block.py)

```python
@T.prim_func
def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
    with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
        rank[0] = T.get_rank()
        num_rank[0] = T.get_num_ranks()
        T.put_block(
            src=T.address_of(src[bx * block_M]),
            dst=T.address_of(dst[bx * block_M]),
            size=block_M,
            dst_pe=rank[0] ^ 1,  # 发送给配对PE
        )
```

**Get Block 原语**: [primitives/example_get_block.py](file:///d:/work/tilescale/examples/distributed/primitives/example_get_block.py)

```python
@T.prim_func
def main(dst: T.Tensor((M), "float32"), src: T.Tensor((M), "float32")):
    with T.Kernel(T.ceildiv(M, block_M), threads=threads) as (bx):
        rank[0] = T.get_rank()
        num_rank[0] = T.get_num_ranks()
        T.get_block(
            src=T.address_of(src[bx * block_M]),
            dst=T.address_of(dst[bx * block_M]),
            size=block_M,
            src_pe=rank[0] ^ 1,  # 从配对PE接收
        )
        T.fence_sys()  # 系统级栅栏
```

---

### 8.3 示例运行方式

```bash
# 设置环境变量
export NVSHMEM_SRC="your_nvshmem_dir"
cd tilelang/distributed
source build_nvshmem.sh

# 安装 pynvshmem
cd pynvshmem && python setup.py install

# 运行示例 (多GPU)
./tilelang/distributed/launch.sh examples/distributed/example_allgather.py

# 或使用 torchrun
torchrun --nproc_per_node=4 examples/distributed/example_allgather.py
```

---

### 8.4 关键 API 总结

| API | 功能 | 级别 |
|-----|------|------|
| `T.get_pe()` | 获取当前 PE ID | Device |
| `T.get_pe_num()` | 获取 PE 总数 | Device |
| `T.get_rank()` | 获取当前 rank | Device |
| `T.get_num_ranks()` | 获取 rank 总数 | Device |
| `T.putmem_nbi_block(dst, src, size, pe)` | 非阻塞 Put | Block |
| `T.getmem_nbi_block(dst, src, size, pe)` | 非阻塞 Get | Block |
| `T.putmem_signal_nbi_block(...)` | 带信号 Put | Block |
| `T.signal_op(addr, value, op, pe)` | 发送信号 | Block |
| `T.signal_wait_until(addr, cmp, value)` | 等待信号 | Block |
| `T.fence()` | 内存栅栏 | Block |
| `T.barrier_all()` | 全局屏障 | Device |
| `T.put_block(src, dst, size, pe)` | Block级Put | Block |
| `T.get_block(src, dst, size, pe)` | Block级Get | Block |

---

*文档生成时间: 2026-03-05*
*分析者: 辉夜*
