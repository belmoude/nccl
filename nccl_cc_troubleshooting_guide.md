# NCCL Initialization Error in Confidential Computing Environment

## Problem Description

When initializing NCCL (NVIDIA Collective Communication Library) in a confidential computing environment, SGLang encounters a bus error (Signal 7) during the shared memory transport setup phase. The error occurs specifically in `shmSendConnect()` function.

## Error Details

```
[2025-08-06 13:08:53 TP0] sglang is using nccl==2.21.5
[2025-08-06 13:08:53 TP1] sglang is using nccl==2.21.5
[VM-32-28-tencentos:10362:0:10362] Caught signal 7 (Bus error: object-specific hardware error)
[VM-32-28-tencentos:10361:0:10361] Caught signal 7 (Bus error: object-specific hardware error)

Backtrace points to:
- ncclAtomicRefCountDecrement<int>()
- shmSendConnect() in /dvs/p4/build/sw/gpgpu/nccl/gitfusion/stable/src/transport/shm.cc:144
- ncclTransportP2pSetup()
- initTransportsRank()
- ncclCommInitRankFunc()
```

## Root Cause Analysis

The error occurs due to several factors specific to confidential computing environments:

1. **Memory Protection**: TEE (Trusted Execution Environment) imposes stricter memory access controls
2. **Shared Memory Restrictions**: VM-based confidential computing may restrict shared memory operations
3. **Atomic Operations**: Hardware atomic operations may be limited in virtualized confidential environments
4. **NCCL Transport Incompatibility**: The shared memory transport in NCCL may not be fully compatible with CC constraints

## Solutions and Workarounds

### Solution 1: Disable Shared Memory Transport

Force NCCL to use network-based transports instead of shared memory:

```bash
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
```

### Solution 2: Use Socket Transport Only

Configure NCCL to use only socket-based communication:

```bash
export NCCL_SOCKET_IFNAME=eth0  # Replace with your network interface
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
```

### Solution 3: Adjust NCCL Buffer Configuration

Reduce memory pressure and avoid problematic memory regions:

```bash
export NCCL_BUFFSIZE=2097152    # 2MB instead of default 4MB
export NCCL_NTHREADS=4          # Reduce thread count
export NCCL_MAX_NCHANNELS=4     # Limit channels
```

### Solution 4: Platform-Specific Configuration

For AMD SEV-SNP environments:
```bash
export NCCL_TOPO_FILE=/dev/null
export NCCL_GRAPH_DUMP_FILE=/dev/null
export NCCL_ALGO=Ring
```

For Intel TDX environments:
```bash
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SOCKET_FAMILY=AF_INET
```

### Solution 5: SGLang-Specific Configuration

When running SGLang in confidential computing:

```python
# In your Python code before initializing SGLang
import os
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SOCKET_IFNAME"] = "eth0"  # Your network interface

# Then initialize SGLang
from sglang import ...
```

Or use command line arguments:
```bash
NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 python -m sglang.launch_server ...
```

## Verification Steps

1. **Check NCCL Configuration**:
```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('NCCL available:', torch.distributed.is_nccl_available())
"
```

2. **Test Basic NCCL Operations**:
```bash
# Create a simple test script
cat > test_nccl.py << 'EOF'
import torch
import torch.distributed as dist
import os

os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

if torch.cuda.is_available():
    dist.init_process_group(backend='nccl', rank=0, world_size=1)
    print("NCCL initialization successful")
    dist.destroy_process_group()
else:
    print("CUDA not available")
EOF

python test_nccl.py
```

3. **Monitor System Resources**:
```bash
# Check shared memory usage
ipcs -m

# Monitor memory pressure
free -h
vmstat 1 5
```

## Advanced Troubleshooting

### Debug NCCL Behavior

Enable detailed NCCL logging:
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
```

### Check Confidential Computing Status

For AMD SEV-SNP:
```bash
# Check if running in SEV-SNP
dmesg | grep -i sev
cat /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | hexdump -C
```

For Intel TDX:
```bash
# Check TDX status
dmesg | grep -i tdx
cpuid | grep -i tdx
```

### Alternative Communication Backends

If NCCL continues to fail, consider alternative backends:

1. **Gloo Backend** (CPU-based):
```python
torch.distributed.init_process_group(backend='gloo')
```

2. **MPI Backend**:
```bash
export CUDA_VISIBLE_DEVICES=0,1
mpirun -np 2 python your_script.py
```

## Best Practices for Confidential Computing

1. **Always disable shared memory transport** in CC environments
2. **Use explicit network interfaces** instead of auto-detection
3. **Reduce buffer sizes** to minimize memory pressure
4. **Test with minimal configurations** first
5. **Monitor system resources** during initialization
6. **Keep NCCL versions updated** for CC compatibility improvements

## Environment Variables Reference

| Variable | Purpose | Recommended Value for CC |
|----------|---------|-------------------------|
| `NCCL_SHM_DISABLE` | Disable shared memory | `1` |
| `NCCL_P2P_DISABLE` | Disable peer-to-peer | `1` |
| `NCCL_NET_GDR_LEVEL` | GPU Direct RDMA level | `0` |
| `NCCL_IB_DISABLE` | Disable InfiniBand | `1` |
| `NCCL_SOCKET_IFNAME` | Network interface | Your interface name |
| `NCCL_BUFFSIZE` | Buffer size | `2097152` (2MB) |
| `NCCL_ALGO` | Algorithm selection | `Ring` or `Tree` |

## Related Issues and References

- NCCL shared memory limitations in virtualized environments
- AMD SEV-SNP memory protection effects on IPC
- Intel TDX restrictions on inter-process communication
- NVIDIA H100 confidential computing constraints

## Support and Updates

This guide addresses NCCL 2.21.5 in confidential computing environments. For updates and additional solutions, monitor:
- NVIDIA NCCL release notes
- Confidential computing documentation updates
- SGLang project issue tracker