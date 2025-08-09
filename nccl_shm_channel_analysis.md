# NCCL中Channel使用SHM传输的数据流和代码流程

## 概述

NCCL (NVIDIA Collective Communication Library) 的 SHM (Shared Memory) 传输是用于同一节点内不同GPU进程间通信的一种高效机制。本文档详细分析了SHM通道的数据流和代码执行流程。

## 核心数据结构

### 1. SHM连接信息结构

```c
struct shmConnectInfo {
  ncclShmIpcDesc_t desc;        // 共享内存IPC描述符
  struct shmBuffInfo buf;       // 缓冲区信息
};

struct shmBuffInfo {
  void *hptr;                   // 主机指针
  void *dptr;                   // 设备指针
};
```

### 2. SHM资源结构

```c
struct shmSendResources {
  struct ncclRecvMem* remHostMem;     // 远程主机内存
  struct ncclRecvMem* devRemHostMem;  // 远程主机内存的设备映射
  ncclShmIpcDesc_t remDesc;           // 远程IPC描述符
  struct ncclSendMem* hostMem;        // 本地主机内存
  struct ncclSendMem* devHostMem;     // 本地主机内存的设备映射
};

struct shmRecvResources {
  struct ncclSendMem* remHostMem;     // 远程主机内存
  struct ncclSendMem* devRemHostMem;  // 远程主机内存的设备映射
  ncclShmIpcDesc_t remDesc;           // 远程IPC描述符
  struct ncclRecvMem* hostMem;        // 本地主机内存
  struct ncclRecvMem* devHostMem;     // 本地主机内存的设备映射
};
```

### 3. 代理信息结构

```c
struct shmProxyInfo {
  struct ncclRecvMem* ceRecvMem;      // 拷贝引擎接收内存
  char* devFifo;                      // 设备FIFO缓冲区
  char* shmFifo;                      // 共享内存FIFO缓冲区
  struct ncclSendMem* sendMem;        // 发送内存
  struct ncclRecvMem* recvMem;        // 接收内存
  
  // 仅用于进度处理
  uint64_t step;                      // 步骤计数
  cudaStream_t stream;                // CUDA流
  cudaEvent_t events[NCCL_STEPS];     // CUDA事件数组
  
  // IPC描述符
  ncclShmIpcDesc_t desc;
};
```

### 4. 连接信息结构

```c
struct ncclConnInfo {
  char *buffs[NCCL_NUM_PROTOCOLS];    // 协议缓冲区
  uint64_t *tail;                     // 尾指针 (本地recv, 远程send)
  uint64_t *head;                     // 头指针 (本地send, 远程recv)
  int stepSize;                       // 步长大小
  struct ncclConnFifo* connFifo;      // 连接FIFO (GPU-代理通信)
  uint64_t step;                      // 当前步骤
  // ... 其他字段
};

struct ncclConnFifo {
  int mode;                           // 模式 (NORMAL/OFFSET/PTR)
  int offset;                         // 偏移量
  ssize_t size;                       // 数据大小
  void* ptr;                          // 指针
};
```

## SHM传输设置流程

### 1. 发送端设置 (shmSendSetup)

```
1. 分配 shmSendResources 结构
2. 根据配置确定共享内存大小:
   - 基本大小: sizeof(struct ncclSendMem)
   - 如果 shmLocality == SHM_SEND_SIDE: 添加所有协议的缓冲区大小
3. 创建共享内存请求 (shmRequest)
4. 通过代理连接进行设置: ncclProxyCallBlocking(..., ncclProxyMsgSetup, ...)
5. 设置主机内存和设备内存指针
```

### 2. 接收端设置 (shmRecvSetup)

```
1. 分配 shmRecvResources 结构
2. 根据配置确定共享内存大小:
   - 基本大小: sizeof(struct ncclRecvMem)
   - 如果 shmLocality == SHM_RECV_SIDE: 添加所有协议的缓冲区大小
3. 创建共享内存请求 (shmRequest)
4. 通过代理连接进行设置: ncclProxyCallBlocking(..., ncclProxyMsgSetup, ...)
5. 设置主机内存和设备内存指针
```

## SHM连接建立流程

### 1. 发送端连接 (shmSendConnect)

```
1. 导入远程共享内存: ncclShmImportShareableBuffer()
2. 设置协议缓冲区指针:
   - 根据 shmLocality 选择缓冲区位置
   - 为每个协议分配缓冲区空间
3. 设置连接信息:
   - conn.tail = 远程主机内存的tail指针
   - conn.head = 本地主机内存的head指针
   - conn.stepSize = 简单协议缓冲区大小 / NCCL_STEPS
4. 如果启用了复制引擎 (useMemcpySend):
   - 通过代理建立连接
   - 更新缓冲区和FIFO指针
```

### 2. 接收端连接 (shmRecvConnect)

```
1. 导入远程共享内存: ncclShmImportShareableBuffer()
2. 设置协议缓冲区指针:
   - 根据 shmLocality 选择缓冲区位置
   - 为每个协议分配缓冲区空间
3. 设置连接信息:
   - conn.head = 远程主机内存的head指针
   - conn.tail = 本地主机内存的tail指针
   - conn.stepSize = 简单协议缓冲区大小 / NCCL_STEPS
4. 如果启用了复制引擎 (useMemcpyRecv):
   - 通过代理建立连接
   - 更新缓冲区和FIFO指针
```

## 数据传输流程

### 1. 直接模式数据流

当未启用CUDA内存拷贝时，GPU直接访问共享内存：

```
发送端:
GPU -> 共享内存缓冲区 -> 接收端GPU

接收端:
共享内存缓冲区 -> GPU
```

**关键点:**
- GPU直接读写共享内存映射区域
- 使用 head/tail 指针进行同步
- 通过 connFifo 传递元数据信息

### 2. 复制引擎模式数据流

当启用CUDA内存拷贝时，通过代理线程进行数据搬移：

#### 发送端复制引擎流程 (shmSendProxyProgress):

```
1. GPU写入数据到本地设备FIFO (devFifo)
2. 代理线程检查数据准备状态 (*recvTail > step)
3. 异步拷贝: devFifo -> shmFifo (设备内存到主机内存)
   cudaMemcpyAsync(shmFifo+offset, devFifo+offset, size, D2H, stream)
4. 记录CUDA事件等待拷贝完成
5. 更新共享内存connFifo[].size，通知接收端
6. 更新接收内存的tail指针
```

#### 接收端复制引擎流程 (shmRecvProxyProgress):

```
1. 代理线程检查共享内存数据准备状态 (*recvTail > step)
2. 异步拷贝: shmFifo -> devFifo (主机内存到设备内存)
   cudaMemcpyAsync(devFifo+offset, shmFifo+offset, size, H2D, stream)
3. 记录CUDA事件等待拷贝完成
4. 更新拷贝引擎接收内存的tail指针，通知GPU
```

## 同步机制

### 1. 生产者-消费者同步

```c
// 发送端 (生产者)
resources->sendMem->head = new_head_value;    // 更新头指针
__sync_synchronize();                         // 内存屏障

// 接收端 (消费者)  
while (*remoteHead <= local_tail) {           // 等待数据可用
  // 自旋等待
}
// 读取数据
resources->recvMem->tail = new_tail_value;    // 更新尾指针
```

### 2. FIFO元数据同步

```c
// 发送端设置FIFO信息
connFifo[slot].size = data_size;              // 设置数据大小
__sync_synchronize();                         // 确保size可见

// 接收端检查FIFO
if (connFifo[slot].size != -1) {              // 检查数据是否准备好
  // 处理数据
  connFifo[slot].size = -1;                   // 重置为未使用状态
}
```

## 内存管理

### 1. 共享内存分配

根据CUDA版本支持不同的分配方式：

**CUDA 12.2+ (cuMem API):**
```c
// 使用CUDA Virtual Memory Management
CUmemGenericAllocationHandle handle;
ncclCuMemHostAlloc(hptr, &handle, size);
// 导出/导入句柄进行跨进程共享
```

**传统模式 (mmap):**
```c
// 使用POSIX共享内存
snprintf(shmPath, sizeof(shmPath), "/dev/shm/nccl-%s", suffix);
ncclShmOpen(shmPath, size, hptr, dptr, create_flag, handle);
```

### 2. 缓冲区布局

```
SHM_SEND_SIDE 布局:
[ncclSendMem][Protocol0_Buffer][Protocol1_Buffer]...[ProtocolN_Buffer]

SHM_RECV_SIDE 布局:
[ncclRecvMem][Protocol0_Buffer][Protocol1_Buffer]...[ProtocolN_Buffer]
```

## 通道初始化流程

### 1. 通道对等体设置

```c
// 在 initChannel() 中:
1. 分配 channel->peers 数组 (每个rank一个)
2. 分配 channel->devPeers 设备端数组
3. 设置引用计数管理
4. 建立与共享资源的连接
```

### 2. 连接器初始化

```c
struct ncclChannelPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];  // 发送连接器
  struct ncclConnector recv[NCCL_MAX_CONNS];  // 接收连接器
  int refCount;                               // 引用计数
};

struct ncclConnector {
  struct ncclProxyConnector proxyConn;        // 代理连接
  struct ncclTransportComm* transportComm;    // 传输通信接口
  void* transportResources;                   // 传输资源 (shmSendResources/shmRecvResources)
  struct ncclConnInfo conn;                   // 连接信息
};
```

## 错误处理和清理

### 1. 资源清理流程

```c
// shmSendFree/shmRecvFree:
1. 关闭IPC描述符: ncclShmIpcClose(&resources->remDesc)
2. 释放传输资源: free(resources)
3. 清空传输资源指针

// shmSendProxyFree/shmRecvProxyFree:
1. 销毁CUDA流: cudaStreamDestroy(stream)
2. 释放设备FIFO: ncclCudaFree(devFifo)
3. 释放主机内存: ncclCudaHostFree(ceRecvMem)
4. 销毁CUDA事件: cudaEventDestroy(events[i])
5. 关闭IPC描述符和释放资源
```

## 性能优化特性

### 1. 本地性配置 (SHM_LOCALITY)

- **SHM_SEND_SIDE (1)**: 缓冲区分配在发送端
- **SHM_RECV_SIDE (2)**: 缓冲区分配在接收端 (默认)

### 2. 内存拷贝模式 (SHM_MEMCPY_MODE)

- **发送端拷贝 (1)**: 使用发送端复制引擎
- **接收端拷贝 (2)**: 使用接收端复制引擎
- **双端拷贝 (3)**: 同时使用两端复制引擎

### 3. 步进式传输

- 数据被分割为 NCCL_STEPS 个步骤
- 支持流水线传输，提高带宽利用率
- 通过环形缓冲区管理多个同时进行的传输

## 调试和监控

### 1. 跟踪日志

```c
TRACE(NCCL_INIT|NCCL_SHM, "Channel %02d : %d[%d] -> %d[%d] via SHM/%s/%s", 
      channelId, myRank, myDev, peerRank, peerDev, 
      useMemcpySend?"CE":"direct", useMemcpyRecv?"CE":"direct");
```

### 2. 参数配置

- `NCCL_SHM_DISABLE=1`: 禁用SHM传输
- `NCCL_SHM_USE_CUDA_MEMCPY=1`: 启用CUDA内存拷贝
- `NCCL_SHM_MEMCPY_MODE`: 设置拷贝模式
- `NCCL_SHM_LOCALITY`: 设置缓冲区本地性

## 总结

NCCL的SHM传输机制通过精心设计的数据结构和同步机制，实现了高效的同节点GPU间通信。关键特性包括：

1. **双模式支持**: 直接访问模式和复制引擎模式
2. **灵活的内存管理**: 支持传统mmap和现代cuMem API
3. **高效的同步机制**: 基于原子操作的生产者-消费者模式
4. **流水线传输**: 通过多步骤环形缓冲区提高吞吐量
5. **丰富的配置选项**: 支持多种性能优化策略

这种设计确保了SHM传输在不同场景下都能提供最佳的性能表现。