#!/usr/bin/env python3
"""
简单的自动化实验脚本
运行FixMatch和MixMatch在40、250、4000标签数据上的实验
"""

import os
import sys
from datetime import datetime


def run_single_experiment(algorithm, n_labeled):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"🚀 开始实验: {algorithm.upper()} with {n_labeled} labeled samples")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    if algorithm == "fixmatch":
        cmd = f"""python train_fixmatch.py \\
    --gpu 0 \\
    --n-labeled {n_labeled} \\
    --batch-size 64 \\
    --total-steps 20000 \\
    --eval-step 1000 \\
    --lr 0.03 \\
    --weight-decay 5e-4 \\
    --lambda-u 1.0 \\
    --threshold 0.95 \\
    --T 1.0 \\
    --mu 7 \\
    --use-ema \\
    --ema-decay 0.999 \\
    --seed 42 \\
    --data-path ./data \\
    --save-path ./saved_models \\
    --log-path ./logs"""
    else:  # mixmatch
        cmd = f"""python train_mixmatch.py \\
    --gpu 0 \\
    --n-labeled {n_labeled} \\
    --batch-size 64 \\
    --total-steps 20000 \\
    --eval-step 1000 \\
    --lr 0.002 \\
    --weight-decay 5e-4 \\
    --lambda-u 75 \\
    --T 0.5 \\
    --alpha 0.75 \\
    --use-ema \\
    --ema-decay 0.999 \\
    --seed 42 \\
    --data-path ./data \\
    --save-path ./saved_models \\
    --log-path ./logs"""

    # 移除换行符，创建单行命令
    cmd = cmd.replace("\\\n", "").replace("    ", " ")

    print(f"📝 执行命令:")
    print(cmd)
    print()

    # 执行命令
    exit_code = os.system(cmd)

    if exit_code == 0:
        print(f"✅ 实验成功: {algorithm.upper()} with {n_labeled} labels")
        return True
    else:
        print(f"❌ 实验失败: {algorithm.upper()} with {n_labeled} labels")
        return False


def main():
    # 创建必要的目录
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    print("🎯 SSL自动化实验脚本")
    print("📊 将运行以下实验:")
    print("   - FixMatch: 40, 250, 4000 labels")
    print("   - MixMatch: 40, 250, 4000 labels")
    print("   - 总共6个实验")
    print()

    # 定义实验配置
    experiments = [
        ("fixmatch", 40),
        ("fixmatch", 250),
        ("fixmatch", 4000),
        ("mixmatch", 40),
        ("mixmatch", 250),
        ("mixmatch", 4000),
    ]

    # 记录结果
    results = []
    failed = []

    # 运行所有实验
    for i, (algorithm, n_labeled) in enumerate(experiments, 1):
        print(f"\n🔄 实验 {i}/6")

        success = run_single_experiment(algorithm, n_labeled)

        if success:
            results.append((algorithm, n_labeled))
            print(f"✅ 完成: {algorithm.upper()} with {n_labeled} labels")
        else:
            failed.append((algorithm, n_labeled))
            print(f"❌ 失败: {algorithm.upper()} with {n_labeled} labels")

    # 打印总结
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    print(f"✅ 成功的实验: {len(results)}/6")
    print(f"❌ 失败的实验: {len(failed)}/6")

    if results:
        print("\n✅ 成功的实验:")
        for algorithm, n_labeled in results:
            print(f"   - {algorithm.upper()}: {n_labeled} labels")

    if failed:
        print("\n❌ 失败的实验:")
        for algorithm, n_labeled in failed:
            print(f"   - {algorithm.upper()}: {n_labeled} labels")

    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"./logs/experiment_summary_{timestamp}.txt"

    with open(summary_file, "w") as f:
        f.write(f"SSL实验总结 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总实验数: 6\n")
        f.write(f"成功: {len(results)}\n")
        f.write(f"失败: {len(failed)}\n\n")

        f.write("成功的实验:\n")
        for algorithm, n_labeled in results:
            f.write(f"  - {algorithm.upper()}: {n_labeled} labels\n")

        f.write("\n失败的实验:\n")
        for algorithm, n_labeled in failed:
            f.write(f"  - {algorithm.upper()}: {n_labeled} labels\n")

    print(f"\n📄 总结已保存到: {summary_file}")

    if failed:
        print("💥 有实验失败!")
        sys.exit(1)
    else:
        print("🎉 所有实验都成功完成!")


if __name__ == "__main__":
    main()
