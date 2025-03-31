import time
import psutil
import cpuinfo
import random

random.seed(42)


def main():
    # 输入验证
    m = int(input("Enter m (512-2048): "))
    n = int(input("Enter n (512-2048): "))
    k = int(input("Enter k (512-2048): "))

    if not all(512 <= x <= 2048 for x in [m, n, k]):
        raise ValueError("All dimensions must be in [512, 2048]")

    def generate_matrix(rows, cols):
        return [[random.random() for _ in range(cols)] for _ in range(rows)]

    # 生成单精度矩阵
    print("\nGenerating matrices...")
    A = generate_matrix(m, n)
    B = generate_matrix(n, k)
    C = [[0.0 for _ in range(k)] for _ in range(m)]

    # 矩阵乘法计时
    print("Calculating matrix product...")
    start_time = time.time()
    for i in range(m):
        for j in range(k):
            for p in range(n):
                C[i][j] += A[i][p] * B[p][j]
    end_time = time.time()
    compute_time = end_time - start_time

    # 性能计算
    total_flops = 2 * m * n * k
    gflops = (total_flops / compute_time) / 1e9

    # 获取CPU信息
    info = cpuinfo.get_cpu_info()
    cpu_freq = psutil.cpu_freq()
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # 修正后的峰值计算
    flags = info.get("flags", [])
    base_freq = cpu_freq.max or 3.0  # GHz

    if "avx512f" in flags:
        flops_per_cycle = 32
    elif "avx2" in flags:
        flops_per_cycle = 16
    else:
        flops_per_cycle = 4

    theoretical_peak = physical_cores * base_freq * 1e6 * flops_per_cycle
    print(f"\nTheoretical Peak: {theoretical_peak / 1e9:.2f} GFLOPS")
    peak_percentage = (gflops / (theoretical_peak / 1e9)) * 100 if theoretical_peak > 0 else 0

    # 输出结果
    print("\nResults:")
    print(f"A: {m}x{n} matrix (sample):\n", [row[:2] for row in A[:2]])
    print(f"\nB: {n}x{k} matrix (sample):\n", [row[:2] for row in B[:2]])
    print(f"\nC: {m}x{k} matrix (sample):\n", [row[:2] for row in C[:2]])
    print(f"\nTime: {compute_time:.4f} seconds")
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"Peak Percentage: {peak_percentage:.2f}%")
    print("\nHardware Info:")
    print(f"CPU: {info['brand_raw']}")
    print(f"Cores: {physical_cores} physical, {logical_cores} logical")
    print(f"Frequency: {base_freq:.1f} GHz")
    print(f"Instruction Set: {', '.join(flags[:10])}...")


if __name__ == "__main__":
    main()
