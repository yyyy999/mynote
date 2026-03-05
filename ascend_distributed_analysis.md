# Ascend NPU 分布式通信能力分析

## 一、概述

本文档分析将 TileScale 分布式功能移植到 Ascend NPU 平台所需了解的关键技术问题，重点关注 HCCL 通信库的能力与 NVSHMEM 的对比。

---

## 二、HCCL 通信原语分析

### 2.1 点对点通信

| API | 类型 | 说明 |
|-----|------|------|
| `HcclSend()` | 同步阻塞 | 发送数据到目标 rank |
| `HcclRecv()` | 同步阻塞 | 从源 rank 接收数据 |
| `HcclIsend()` | 异步非阻塞 | 异步发送 |
| `HcclIrecv()` | 异步非阻塞 | 异步接收 |
| `HcclBatchSendRecv()` | 批量异步 | 批量点对点通信 |

**关键约束**（来自华为官方文档）：

> HcclSend 与 HcclRecv 接口采用同步调用方式，且必须配对使用。即一个进程调用 HcclSend 接口后，需要等到与之配对的 HcclRecv 接口接收数据后，才可以进行下一个接口调用。

```cpp
// HCCL 点对点通信示例
HcclResult HcclSend(const void* sendBuf, uint64_t count, 
                    HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStream stream);

HcclResult HcclRecv(void* recvBuf, uint64_t count, 
                    HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStream stream);
```

### 2.2 集合通信

| 操作 | API | 说明 |
|------|-----|------|
| AllReduce | `HcclAllReduce()` | 全局规约 |
| AllGather | `HcclAllGather()` | 全局收集 |
| ReduceScatter | `HcclReduceScatter()` | 规约散射 |
| Broadcast | `HcclBroadcast()` | 广播 |
| AlltoAll | `HcclAlltoAll()` | 全交换 |
| Barrier | `HcclBarrier()` | 屏障同步 |

### 2.3 HCCL 不支持的原语

| NVSHMEM 原语 | HCCL 状态 | 影响 |
|--------------|-----------|------|
| `nvshmem_putmem()` | ❌ 不支持 | 无法单边写入远程内存 |
| `nvshmem_getmem()` | ❌ 不支持 | 无法单边读取远程内存 |
| `nvshmem_put_signal()` | ❌ 不支持 | 无法带信号写入 |
| `nvshmem_wait_until()` | ❌ 不支持 | 无法细粒度等待 |
| `nvshmem_atomic_*` | ❌ 不支持 | 无跨 NPU 原子操作 |
| 对称内存 | ❌ 不支持 | 无全局统一地址空间 |

---

## 三、NVSHMEM vs HCCL 语义对比

### 3.1 通信模型对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                    通信模型对比                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【NVSHMEM - PGAS 模型】                                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  特点: 单边操作 + 对称内存                                 │       │
│  │                                                          │       │
│  │  PE 0:                      PE 1:                        │       │
│  │  ┌─────────┐               ┌─────────┐                  │       │
│  │  │ src     │ ── put ────→  │ dest    │                  │       │
│  │  │ (本地)  │               │ (远程)  │                  │       │
│  │  └─────────┘               └─────────┘                  │       │
│  │                                                          │       │
│  │  特点: PE 1 不需要参与，PE 0 直接写入远程内存             │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
│  【HCCL - MPI-like 模型】                                           │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  特点: 双边操作 + 显式配对                                 │       │
│  │                                                          │       │
│  │  PE 0 (发送方):             PE 1 (接收方):               │       │
│  │  ┌─────────┐               ┌─────────┐                  │       │
│  │  │ HcclSend│ ───────────→  │ HcclRecv│                  │       │
│  │  │ (阻塞)  │               │ (阻塞)  │                  │       │
│  │  └─────────┘               └─────────┘                  │       │
│  │                                                          │       │
│  │  特点: 必须配对调用，双方都需要参与                        │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 API 对照表

| 功能 | NVSHMEM | HCCL | 可行性 |
|------|---------|------|--------|
| 获取 PE ID | `nvshmem_my_pe()` | `HcclGetRankId()` | ✅ 直接映射 |
| 获取 PE 总数 | `nvshmem_n_pes()` | `HcclGetRankSize()` | ✅ 直接映射 |
| 全局屏障 | `nvshmem_barrier_all()` | `HcclBarrier()` | ✅ 直接映射 |
| 全局收集 | `nvshmem_allgather()` | `HcclAllGather()` | ✅ 直接映射 |
| 全局规约 | `nvshmem_allreduce()` | `HcclAllReduce()` | ✅ 直接映射 |
| 单边 Put | `nvshmem_putmem()` | ❌ 无 | ⚠️ 需要替代方案 |
| 单边 Get | `nvshmem_getmem()` | ❌ 无 | ⚠️ 需要替代方案 |
| 非阻塞 Put | `nvshmem_putmem_nbi()` | ❌ 无 | ⚠️ 需要替代方案 |
| Signal 写入 | `nvshmem_put_signal()` | ❌ 无 | ⚠️ 需要替代方案 |
| Signal 等待 | `nvshmem_wait_until()` | ❌ 无 | ⚠️ 需要替代方案 |

---

## 四、昇腾超节点与内存统一编址

### 4.1 超节点架构

根据华为公开资料，昇腾 384 超节点具备以下能力：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    昇腾超节点架构                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  核心特性:                                                          │
│  ├── 大带宽: 通信带宽较传统架构提升 15 倍                            │
│  ├── 低时延: RTT 从 7μs 降至 3μs                                    │
│  └── 内存统一编址: 全局唯一地址空间                                  │
│                                                                     │
│  硬件基础:                                                          │
│  ├── HCCS (Huawei Cache Coherent System): 类似 NVLink              │
│  ├── 灵衢 UB 协议: 支持缓存一致性的互联协议                         │
│  └── RoCE v2: 跨节点 RDMA 通信                                      │
│                                                                     │
│  软件能力:                                                          │
│  ├── 通过 load/store 指令直接访问远端内存                           │
│  ├── 支持"One NPU"开发范式                                          │
│  └── 消除"序列化-网络传输-反序列化"开销                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 内存语义 vs 消息语义

| 特性 | 消息语义 (传统) | 内存语义 (超节点) |
|------|-----------------|-------------------|
| 通信方式 | Send/Recv | Load/Store |
| 数据打包 | 需要 | 不需要 |
| 协议开销 | 高 (封装/解封) | 低 (直接访问) |
| CPU 参与 | 需要 | 不需要 |
| 延迟 | 毫秒级 | 微秒级 |
| 代表协议 | TCP/IP, RoCE | NVLink, 灵衢 UB |

### 4.3 关键发现

**昇腾超节点支持内存统一编址，但 HCCL 库层面尚未暴露单边操作 API。**

这意味着：
1. **硬件层面**：昇腾超节点具备单边通信的硬件基础
2. **软件层面**：HCCL 目前只提供双边通信 API
3. **移植策略**：可能需要：
   - 等待华为开放更底层的内存访问 API
   - 或通过 HCCL 的 Send/Recv 模拟单边操作
   - 或重新设计分布式算法

---

## 五、TileScale 原语映射方案

### 5.1 直接映射（可行）

| TileScale 原语 | HCCL 映射 | 代码示例 |
|----------------|-----------|----------|
| `T.get_pe()` | `HcclGetRankId()` | `int rank = HcclGetRankId(comm);` |
| `T.get_pe_num()` | `HcclGetRankSize()` | `int size = HcclGetRankSize(comm);` |
| `T.barrier_all()` | `HcclBarrier()` | `HcclBarrier(comm, stream);` |
| AllGather | `HcclAllGather()` | `HcclAllGather(sendbuf, recvbuf, count, ...);` |
| AllReduce | `HcclAllReduce()` | `HcclAllReduce(sendbuf, recvbuf, count, HCCL_SUM, ...);` |

### 5.2 需要替代方案

#### 5.2.1 Put 操作模拟

```cpp
// NVSHMEM 方式 (单边)
nvshmem_putmem_nbi_block(dest, src, size, peer);  // PE 0 直接写入 PE 1

// HCCL 方式 (双边配对)
if (my_rank == sender) {
    HcclSend(src, size, datatype, peer, comm, stream);
} else {
    HcclRecv(dst, size, datatype, sender, comm, stream);
}
```

**问题**：双边通信需要接收方主动调用 Recv，无法实现真正的"单边"语义。

#### 5.2.2 Ring AllGather 算法适配

```python
# NVSHMEM 版本 (单边 put)
for step in range(npes - 1):
    peer = (mype + step + 1) % npes
    putmem_nbi(dest, src, size, peer)  # 单边发送
    barrier_all()

# HCCL 版本方案 A: 使用内置 AllGather
HcclAllGather(sendbuf, recvbuf, size, comm, stream)

# HCCL 版本方案 B: Send/Recv 配对
for step in range(npes - 1):
    send_peer = (mype + step + 1) % npes
    recv_peer = (mype - step - 1) % npes
    HcclSend(send_buf, size, datatype, send_peer, comm, stream)
    HcclRecv(recv_buf, size, datatype, recv_peer, comm, stream)
```

#### 5.2.3 Signal/Wait 模拟

```cpp
// NVSHMEM 方式
nvshmem_put_signal(dest, src, size, signal_addr, value, pe);
nvshmem_wait_until(signal, EQ, expected);

// HCCL 方案: 使用 Barrier 替代 (粗粒度)
HcclBarrier(comm, stream);

// 或使用小数据 AllReduce 模拟
uint64_t signal = 1;
HcclAllReduce(&signal, &global_signal, 1, HCCL_SUM, comm, stream);
```

### 5.3 算法层面重新设计

对于依赖单边操作的算法，需要重新设计：

| 算法 | NVSHMEM 实现 | HCCL 替代方案 |
|------|--------------|---------------|
| Ring AllGather | 单边 put | 使用 `HcclAllGather` 或 Send/Recv 配对 |
| All2All (MoE) | 单边 put + signal | 使用 `HcclAlltoAll` 或 Send/Recv 配对 |
| SUMMA | 广播 + 本地计算 | 使用 `HcclBroadcast` |
| Cannon | 环形移位 | Send/Recv 配对 |

---

## 六、NCCL 单边操作参考

值得注意的是，NCCL 从较新版本开始也支持单边操作：

```cpp
// NCCL One-Sided RMA Operations (NCCL 2.9+)

// 注册内存窗口
ncclWindow_t win;
ncclCommWindowRegister(comm, buffer, size, &win);

// 单边写入 + Signal
ncclPutSignal(localbuff, count, datatype, peer, win, offset, 
              sigIdx, ctx, flags, comm, stream);

// 等待 Signal
ncclWaitSignal(nDesc, signalDescs, comm, stream);
```

这表明 NVIDIA 也在向单边通信演进。如果华为后续在 HCCL 中开放类似能力，移植工作将更加顺畅。

---

## 七、结论与建议

### 7.1 当前状态总结

| 维度 | 状态 | 说明 |
|------|------|------|
| 硬件能力 | ✅ 支持 | 昇腾超节点支持内存统一编址 |
| HCCL 集合通信 | ✅ 完整 | AllReduce/AllGather/Broadcast 等完整支持 |
| HCCL 点对点 | ⚠️ 双边 | 只有 Send/Recv，无单边操作 |
| Signal/Wait | ❌ 不支持 | 无细粒度同步机制 |
| 对称内存 | ❌ 不支持 | HCCL 层面无对称内存抽象 |

### 7.2 移植策略建议

1. **短期方案**：
   - 使用 HCCL 集合通信替代单边操作
   - 使用 Send/Recv 配对模拟点对点通信
   - 使用 Barrier 替代 Signal/Wait

2. **中期方案**：
   - 关注华为是否在 HCCL 中开放单边操作 API
   - 关注昇腾超节点的底层内存访问接口

3. **长期方案**：
   - 如果华为开放类似 NVSHMEM 的单边通信库，可直接映射
   - 否则需要基于昇腾超节点的内存统一编址能力自行实现

### 7.3 需要进一步调研的问题

1. 昇腾超节点的内存统一编址是否有用户态 API？
2. 是否可以通过 Ascend C++ 直接访问远端内存？
3. 华为是否有类似 NVSHMEM 的单边通信库规划？

---

## 八、参考资料

1. [HCCL API 文档 - HcclSend/HcclRecv](https://www.hiascend.com/doc_center/source/zh/canncommercial/63RC1/modeldev/tfmigr1/tfmigr_hcclopbase_0016.html)
2. [HCCL 概述](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devguide/hccl/hcclug/hcclug_000001.html)
3. [深度学习的分布式训练与集合通信](https://www.hiascend.com/developer/techArticles/20241111-1)
4. [NCCL Point-to-Point Communication Functions](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2293/user-guide/docs/api/p2p.html)
5. [NVSHMEM Remote Memory Access (RMA)](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/rma.html)
6. [昇腾超节点技术分析](https://developer.aliyun.com/article/1709969)
7. [华为昇腾架构介绍](https://www.xiexianbin.cn/hardware/huawei-ascend/index.html)
