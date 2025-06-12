#!/usr/bin/env python3
"""
运行TorchSSL实验的脚本
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime


def run_torchssl_experiment(algorithm, num_labels, gpu_id=0):
    """运行单个TorchSSL实验"""

    # 设置实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"torchssl_{algorithm}_{num_labels}labels_{timestamp}"

    # 设置保存目录
    save_dir = f"./results/torchssl_{algorithm}_{num_labels}labels"
    os.makedirs(save_dir, exist_ok=True)

    # 配置文件路径
    config_file = f"TorchSSL/config/{algorithm}/{algorithm}_cifar10_{num_labels}_0.yaml"

    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False

    # 构建命令
    cmd = [
        "python",
        f"TorchSSL/{algorithm}.py",
        "--c",
        config_file,
        "--save_dir",
        save_dir,
        "--save_name",
        f"{algorithm}_cifar10_{num_labels}_0",
        "--overwrite=True",
        "--gpu",
        str(gpu_id),
    ]

    print(f"🚀 Running experiment: {exp_name}")
    print(f"📁 Save directory: {save_dir}")
    print(f"⚙️  Config file: {config_file}")
    print(f"💻 Command: {' '.join(cmd)}")
    print("=" * 80)

    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 运行实验
    start_time = time.time()
    try:
        result = subprocess.run(cmd, cwd=os.getcwd(), env=env, timeout=7200)  # 2小时超时

        end_time = time.time()
        duration = end_time - start_time

        # 保存结果摘要
        result_file = os.path.join(save_dir, "experiment_summary.json")
        with open(result_file, "w") as f:
            json.dump(
                {
                    "experiment_name": exp_name,
                    "algorithm": algorithm,
                    "num_labels": num_labels,
                    "config_file": config_file,
                    "command": " ".join(cmd),
                    "return_code": result.returncode,
                    "duration_seconds": duration,
                    "timestamp": timestamp,
                    "save_dir": save_dir,
                },
                f,
                indent=2,
            )

        if result.returncode == 0:
            print(f"✅ Experiment {exp_name} completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ Experiment {exp_name} failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {exp_name} timed out after 2 hours")
        return False
    except Exception as e:
        print(f"💥 Experiment {exp_name} failed with exception: {e}")
        return False


def run_all_torchssl_experiments():
    """运行所有TorchSSL实验"""

    algorithms = ["fixmatch", "mixmatch"]
    num_labels_list = [40, 250, 4000]

    results = {}
    total_experiments = len(algorithms) * len(num_labels_list)
    current_exp = 0

    print(f"🎯 Starting {total_experiments} TorchSSL experiments")
    print("=" * 80)

    for algorithm in algorithms:
        results[algorithm] = {}
        for num_labels in num_labels_list:
            current_exp += 1
            print(f"\n📊 Experiment {current_exp}/{total_experiments}")
            print(f"🔬 Algorithm: {algorithm.upper()}")
            print(f"🏷️  Labels: {num_labels}")
            print("-" * 40)

            success = run_torchssl_experiment(algorithm, num_labels)
            results[algorithm][num_labels] = success

            if success:
                print(f"✅ {algorithm} with {num_labels} labels: SUCCESS")
            else:
                print(f"❌ {algorithm} with {num_labels} labels: FAILED")

    # 保存总结果
    summary_file = f"./results/torchssl_experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)

    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    # 打印总结
    print(f"\n{'='*80}")
    print("📋 TORCHSSL EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    success_count = 0
    for algorithm in algorithms:
        print(f"\n🔬 {algorithm.upper()}:")
        for num_labels in num_labels_list:
            status = "✅" if results[algorithm][num_labels] else "❌"
            print(f"   {num_labels:4d} labels: {status}")
            if results[algorithm][num_labels]:
                success_count += 1

    print(f"\n📊 Overall: {success_count}/{total_experiments} experiments successful")
    print(f"📄 Detailed results saved to: {summary_file}")

    return results


def run_single_torchssl_experiment():
    """运行单个TorchSSL实验（用于测试）"""
    parser = argparse.ArgumentParser(description="Run a single TorchSSL experiment")
    parser.add_argument("--algorithm", choices=["fixmatch", "mixmatch"], required=True)
    parser.add_argument("--num_labels", type=int, choices=[40, 250, 4000], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)

    args = parser.parse_args()

    success = run_torchssl_experiment(args.algorithm, args.num_labels, args.gpu_id)

    if success:
        print("🎉 Experiment completed successfully!")
    else:
        print("💥 Experiment failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run TorchSSL experiments")
    parser.add_argument(
        "--mode", choices=["single", "all"], default="all", help="Run single experiment or all experiments"
    )

    args, remaining = parser.parse_known_args()

    if args.mode == "single":
        # 将剩余参数传递给单个实验函数
        sys.argv = [sys.argv[0]] + remaining
        run_single_torchssl_experiment()
    else:
        run_all_torchssl_experiments()


if __name__ == "__main__":
    main()
