#!/bin/bash

echo "=== 混合CC环境验证脚本 ==="
echo "检查CPU CC状态和GPU CC状态"
echo

# 检查CPU CC状态
echo "1. 检查CPU机密计算状态："
if dmesg | grep -qi "sev.*active"; then
    echo "   ✓ AMD SEV-SNP 已启用"
    dmesg | grep -i sev | tail -3
elif dmesg | grep -qi "tdx.*initialized"; then
    echo "   ✓ Intel TDX 已启用" 
    dmesg | grep -i tdx | tail -3
else
    echo "   ⚠ 未检测到CPU CC特性"
fi

echo

# 检查GPU CC状态
echo "2. 检查GPU机密计算状态："
if nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -qi "h100"; then
    echo "   H100 GPU detected, checking CC mode..."
    # 检查是否处于CC模式
    nvidia-smi -q | grep -i confidential || echo "   ✗ GPU未启用CC模式"
else
    echo "   ✗ 非H100 GPU，不支持机密计算"
    nvidia-smi --query-gpu=name --format=csv,noheader,nounits
fi

echo

# 检查共享内存状态
echo "3. 检查共享内存环境："
echo "   当前共享内存段："
ipcs -m | grep -v "^key" | wc -l
echo "   /dev/shm 挂载状态："
mount | grep shm
echo "   共享内存限制："
cat /proc/sys/kernel/shmmax

echo

# 检查内存保护状态
echo "4. 检查内存保护状态："
echo "   当前进程内存映射特征："
cat /proc/self/status | grep -E "VmRSS|VmSize|VmStk"
echo "   内存加密状态："
grep -r encrypt /proc/iomem 2>/dev/null | head -3 || echo "   无特殊内存加密检测到"

echo

# 测试NCCL环境变量影响
echo "5. NCCL环境变量测试："
echo "   当前NCCL相关环境变量："
env | grep NCCL | sort

if [ -z "$NCCL_SHM_DISABLE" ]; then
    echo "   ⚠ NCCL_SHM_DISABLE 未设置 - 可能导致bus error"
else
    echo "   ✓ NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE - 应该可以避免问题"
fi

echo

# 网络接口检查
echo "6. 网络接口状态（NCCL网络传输用）："
ip addr show | grep -E "inet.*scope global" | head -5

echo
echo "=== 建议配置 ==="
echo "基于检测结果，建议设置以下环境变量："
echo "export NCCL_SHM_DISABLE=1"
echo "export NCCL_P2P_DISABLE=1" 
echo "export NCCL_NET_GDR_LEVEL=0"
echo "export NCCL_SOCKET_IFNAME=$(ip route | grep default | awk '{print $5}' | head -1)"
echo

echo "=== 快速测试命令 ==="
echo "测试NCCL初始化："
cat << 'EOF'
# 创建测试脚本
cat > test_nccl_cc.py << 'PYEOF'
import torch
import os

print("=== NCCL CC环境测试 ===")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# 设置CC环境的NCCL配置
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

if torch.cuda.is_available():
    try:
        # 尝试初始化NCCL
        torch.distributed.init_process_group(
            backend='nccl', 
            rank=0, 
            world_size=1,
            init_method='tcp://localhost:29500'
        )
        print("✓ NCCL initialization successful!")
        torch.distributed.destroy_process_group()
    except Exception as e:
        print(f"✗ NCCL initialization failed: {e}")
else:
    print("⚠ CUDA not available")
PYEOF

# 运行测试
python test_nccl_cc.py
EOF