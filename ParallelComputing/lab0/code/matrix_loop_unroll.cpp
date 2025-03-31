#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

using namespace std;

// 生成随机矩阵
vector<vector<double>> generate_matrix(int rows, int cols) {
    vector<vector<double>> matrix(rows, vector<double>(cols));
    random_device rd;
    mt19937 gen(42);
    uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

// 打印矩阵样本
void print_matrix_sample(const vector<vector<double>>& matrix, int rows, int cols) {
    for (int i = 0; i < min(rows, 2); ++i) {
        for (int j = 0; j < min(cols, 2); ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

int main() {
    // 输入验证
    int m, n, k;
    cout << "Enter m (512-2048): ";
    cin >> m;
    cout << "Enter n (512-2048): ";
    cin >> n;
    cout << "Enter k (512-2048): ";
    cin >> k;

    if (m < 512 || m > 2048 || n < 512 || n > 2048 || k < 512 || k > 2048) {
        cerr << "All dimensions must be in [512, 2048]" << endl;
        return 1;
    }

    // 生成单精度矩阵
    cout << "\nGenerating matrices..." << endl;
    vector<vector<double>> A = generate_matrix(m, n);
    vector<vector<double>> B = generate_matrix(n, k);
    vector<vector<double>> C(m, vector<double>(k, 0.0));

    // 矩阵乘法计时
    cout << "Calculating matrix product..." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    for (int i = 0; i < m; ++i) {
        for (int p = 0; p < n; ++p) { // 调整循环顺序
            int j = 0;
            for (; j < k - 3; j+=4) {
                C[i][j] += A[i][p] * B[p][j];
                C[i][j+1] += A[i][p] * B[p][j+1];
                C[i][j+2] += A[i][p] * B[p][j+2];
                C[i][j+3] += A[i][p] * B[p][j+3];
            }
            for (; j < k; ++j) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    double compute_time = chrono::duration_cast<chrono::duration<double>>(end_time - start_time).count();

    // 性能计算
    long long total_flops = 2LL * m * n * k;
    double gflops = (total_flops / compute_time) / 1e9;

    // 峰值性能百分比
    double theoretical_peak = 1267.20; // GFLOPS
    double peak_percentage = (gflops / theoretical_peak) * 100;

    // 输出结果
    cout << "\nResults:" << endl;
    cout << "A: " << m << "x" << n << " matrix (sample):" << endl;
    print_matrix_sample(A, m, n);

    cout << "\nB: " << n << "x" << k << " matrix (sample):" << endl;
    print_matrix_sample(B, n, k);

    cout << "\nC: " << m << "x" << k << " matrix (sample):" << endl;
    print_matrix_sample(C, m, k);

    cout << "\nTime: " << fixed << setprecision(4) << compute_time << " seconds" << endl;
    cout << "Performance: " << fixed << setprecision(2) << gflops << " GFLOPS" << endl;
    cout << "Peak Percentage: " << fixed << setprecision(2) << peak_percentage << "%" << endl;

    return 0;
}