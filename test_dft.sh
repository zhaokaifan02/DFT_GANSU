#!/bin/bash
# DFT 测试脚本

echo "======================================"
echo "DFT 能量计算测试"
echo "======================================"
echo ""

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行 DFT 计算
echo "运行命令: ./HF_main -m RHF -g ../basis/sto-3g.gbs -x ../xyz/H2O.xyz --eri_method dft"
echo ""

./HF_main -m RHF -g ../basis/sto-3g.gbs -x ../xyz/H2O.xyz --eri_method dft

echo ""
echo "======================================"
echo "测试完成！"
echo "======================================"
echo ""
echo "预期结果："
echo "- DFT 能量应该使用 Vxc（交换-相关势）"
echo "- Fock 矩阵公式：F = H + 2J + Vxc"
echo "- 能量不应该再有 ~8 au 的偏差"
echo ""
echo "对比测试（可选）："
echo "1. HF 方法: ./HF_main -m RHF -g ../basis/sto-3g.gbs -x ../xyz/H2O.xyz --eri_method stored"
echo "2. DFT 方法: ./HF_main -m RHF -g ../basis/sto-3g.gbs -x ../xyz/H2O.xyz --eri_method dft"
echo ""
