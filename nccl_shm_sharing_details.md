# NCCL SHM共享内存详解：谁和谁在共享？

## 共享内存的参与者

在NCCL SHM（Shared Memory）机制中，共享内存主要在以下实体之间进行：

### 1. 主要共享对象：GPU进程间的数据传输

```
同一台机器上的多个GPU进程：
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GPU 0 进程    │    │   GPU 1 进程    │    │   GPU 2 进程    │
│   (rank 0)      │    │   (rank 1)      │    │   (rank 2)      │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │  NCCL     │  │    │  │  NCCL     │  │    │  │  NCCL     │  │
│  │  Comm     │  │◄──►│  │  Comm     │  │◄──►│  │  Comm     │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │      共享内存区域         │
                    │    /dev/shm/nccl-*      │
                    │                         │
                    │  ┌─────────────────┐    │
                    │  │ 控制信息        │    │
                    │  │ - 引用计数      │    │
                    │  │ - 同步标志      │    │
                    │  │ - 状态信息      │    │
                    │  └─────────────────┘    │
                    │  ┌─────────────────┐    │
                    │  │ 数据缓冲区      │    │
                    │  │ - 消息队列      │    │
                    │  │ - 数据暂存      │    │
                    │  │ - 元数据        │    │
                    │  └─────────────────┘    │
                    └─────────────────────────┘
```

### 2. 具体的进程类型

#### A. 多进程分布式训练场景
```python
# 典型的PyTorch DDP (DistributedDataParallel) 场景
# 启动命令：torchrun --nproc_per_node=4 train.py

进程结构：
- 主进程 (launcher): python -m torch.distributed.launch
  └── 子进程 rank 0: python train.py --local_rank=0  (使用GPU 0)
  └── 子进程 rank 1: python train.py --local_rank=1  (使用GPU 1) 
  └── 子进程 rank 2: python train.py --local_rank=2  (使用GPU 2)
  └── 子进程 rank 3: python train.py --local_rank=3  (使用GPU 3)

# 这4个子进程通过NCCL SHM共享内存进行通信
```

#### B. SGLang多GPU场景
```python
# SGLang启动多GPU服务器
# 命令：python -m sglang.launch_server --model-path xxx --tp-size 2

进程结构：
- SGLang主进程: 协调和管理
  └── TP进程 0: 处理模型的第一部分 (GPU 0)
  └── TP进程 1: 处理模型的第二部分 (GPU 1)

# TP(Tensor Parallel)进程间通过NCCL SHM交换张量数据
```

### 3. 共享内存的具体内容

#### 内存布局结构
```c
// NCCL共享内存段的典型结构
struct ncclShmem {
    // 控制结构
    struct {
        volatile int refCount;        // 引用计数 ← bus error发生点
        volatile int initialized;     // 初始化标志
        volatile int generation;      // 版本号
        pthread_mutex_t mutex;        // 互斥锁
        pthread_cond_t cond;         // 条件变量
    } control;
    
    // 通信缓冲区
    struct {
        char sendBuff[NCCL_SHM_SIZE]; // 发送缓冲区
        char recvBuff[NCCL_SHM_SIZE]; // 接收缓冲区
        volatile int sendHead;         // 发送队列头
        volatile int sendTail;         // 发送队列尾
        volatile int recvHead;         // 接收队列头
        volatile int recvTail;         // 接收队列尾
    } buffers;
    
    // 同步原语
    struct {
        volatile int barrierCount;     // 屏障计数
        volatile int barrierGeneration;// 屏障版本
        sem_t semaphores[MAX_RANKS];   // 信号量数组
    } sync;
};
```

### 4. 共享内存的生命周期

#### 创建阶段
```bash
# 当第一个NCCL进程启动时
ls -la /dev/shm/
# 空的，没有nccl相关文件

# 第一个进程调用ncclCommInitRank()
# 创建共享内存段
/dev/shm/nccl-shm-<pid>-<rank>  # 出现共享内存文件
```

#### 使用阶段
```c
// 每个GPU进程的操作流程
1. 打开已存在的共享内存段
   fd = shm_open("/nccl-shm-12345-0", O_RDWR, 0666);

2. 映射到进程地址空间
   void* shmPtr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);

3. 原子操作增加引用计数 ← 问题点
   __sync_add_and_fetch(&((struct ncclShmem*)shmPtr)->control.refCount, 1);

4. 使用共享缓冲区进行通信
   memcpy(shmPtr->buffers.sendBuff, data, size);
   
5. 同步等待其他进程
   __sync_synchronize();
```

#### 清理阶段
```c
// 最后一个进程退出时
__sync_sub_and_fetch(&refCount, 1);  // ← bus error发生的具体位置
if (refCount == 0) {
    shm_unlink("/nccl-shm-12345-0");  // 删除共享内存段
}
```

### 5. 在您的混合CC环境中的问题

#### 进程安全域分离
```
TEE域中的进程:
┌─────────────────────────────────┐
│ SGLang TP进程 0 (rank 0)        │ ← 运行在CPU CC模式中
│ - 内存加密                      │
│ - 完整性保护                    │  
│ - 严格访问控制                  │
└─────────────────────────────────┘
              │
              │ 尝试与下方进程共享内存
              ▼
┌─────────────────────────────────┐
│ SGLang TP进程 1 (rank 1)        │ ← 运行在普通模式中
│ - 普通内存                      │
│ - 无特殊保护                    │
│ - 标准访问权限                  │
└─────────────────────────────────┘

问题：两个进程位于不同安全域，无法安全共享内存页面
```

#### 具体冲突点
```c
// rank 0进程 (TEE中) 创建共享内存
fd = shm_open("/nccl-shm-1234", O_CREAT|O_RDWR, 0666);
// 内存页面被标记为TEE保护

// rank 1进程 (普通域) 尝试访问
fd = shm_open("/nccl-shm-1234", O_RDWR, 0666);        // 可能成功
ptr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0); // 可能成功

// 但是原子操作失败！
__sync_add_and_fetch(&((struct ncclShmem*)ptr)->control.refCount, 1);
// ↑ 硬件级安全检查阻止跨域原子操作 → bus error (signal 7)
```

### 6. 为什么禁用SHM有效

#### 替代通信方式
```
禁用SHM后的通信路径：
┌─────────────────┐    ┌─────────────────┐
│ TP进程 0 (TEE)  │    │ TP进程 1 (普通) │
│                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │   Socket    │ │◄──►│ │   Socket    │ │
│ │  (网络栈)   │ │    │ │  (网络栈)   │ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘
         │                       │
         └───── TCP/IP 网络 ──────┘
              (通过内核网络层)
              无直接内存共享
```

#### 网络传输的优势
- **安全域隔离友好**：通过内核网络栈，不直接共享内存页面
- **跨域兼容**：TEE和普通进程都可以使用标准socket API
- **无原子操作冲突**：网络传输不需要跨进程原子操作

## 总结

NCCL SHM中的共享内存是在**同一节点上的多个GPU进程之间**共享的，具体包括：

1. **共享对象**：分布式训练中的不同rank进程（如SGLang的TP进程）
2. **共享内容**：控制信息、数据缓冲区、同步原语  
3. **共享目的**：高效的进程间通信，避免网络开销
4. **问题根源**：在混合CC环境中，TEE进程和普通进程无法安全共享内存页面
5. **解决方案**：禁用SHM，使用网络传输替代直接内存共享

这就是为什么`NCCL_SHM_DISABLE=1`能够解决您的bus error问题的根本原因。