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

// 矩阵加法
void add_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// 矩阵减法
void subtract_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B, vector<vector<double>>& C) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Strassen算法实现
vector<vector<double>> strassen(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();

    // 基本情况：使用传统矩阵乘法
    if (n <= 128) {
        vector<vector<double>> C(n, vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // 分治法
    int half = n / 2;
    vector<vector<double>> A11(half, vector<double>(half));
    vector<vector<double>> A12(half, vector<double>(half));
    vector<vector<double>> A21(half, vector<double>(half));
    vector<vector<double>> A22(half, vector<double>(half));
    vector<vector<double>> B11(half, vector<double>(half));
    vector<vector<double>> B12(half, vector<double>(half));
    vector<vector<double>> B21(half, vector<double>(half));
    vector<vector<double>> B22(half, vector<double>(half));

    // 分割矩阵
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + half];
            B21[i][j] = B[i + half][j];
            B22[i][j] = B[i + half][j + half];
        }
    }

    // 创建临时矩阵用于存储中间结果
    vector<vector<double>> temp1(half, vector<double>(half));
    vector<vector<double>> temp2(half, vector<double>(half));

    // 计算中间结果
    subtract_matrices(B12, B22, temp1);
    vector<vector<double>> M1 = strassen(A11, temp1);

    add_matrices(A11, A12, temp1);
    vector<vector<double>> M2 = strassen(temp1, B22);

    add_matrices(A21, A22, temp1);
    vector<vector<double>> M3 = strassen(temp1, B11);

    subtract_matrices(B21, B11, temp1);
    vector<vector<double>> M4 = strassen(A22, temp1);

    add_matrices(A11, A22, temp1);
    add_matrices(B11, B22, temp2);
    vector<vector<double>> M5 = strassen(temp1, temp2);

    subtract_matrices(A12, A22, temp1);
    add_matrices(B21, B22, temp2);
    vector<vector<double>> M6 = strassen(temp1, temp2);

    subtract_matrices(A11, A21, temp1);
    add_matrices(B11, B12, temp2);
    vector<vector<double>> M7 = strassen(temp1, temp2);

    // 计算结果矩阵的四个部分
    vector<vector<double>> C11(half, vector<double>(half));
    vector<vector<double>> C12(half, vector<double>(half));
    vector<vector<double>> C21(half, vector<double>(half));
    vector<vector<double>> C22(half, vector<double>(half));

    // 计算 C11 = M5 + M4 - M2 + M6
    add_matrices(M5, M4, C11);
    subtract_matrices(C11, M2, C11);
    add_matrices(C11, M6, C11);

    // 计算 C12 = M1 + M2
    add_matrices(M1, M2, C12);

    // 计算 C21 = M3 + M4
    add_matrices(M3, M4, C21);

    // 计算 C22 = M5 + M1 - M3 - M7
    add_matrices(M5, M1, C22);
    subtract_matrices(C22, M3, C22);
    subtract_matrices(C22, M7, C22);

    // 合并结果矩阵
    vector<vector<double>> C(n, vector<double>(n));
    for (int i = 0; i < half; ++i) {
        for (int j = 0; j < half; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + half] = C12[i][j];
            C[i + half][j] = C21[i][j];
            C[i + half][j + half] = C22[i][j];
        }
    }

    return C;
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

    // 矩阵乘法计时
    cout << "Calculating matrix product..." << endl;
    auto start_time = chrono::high_resolution_clock::now();

    // 使用Strassen算法
    vector<vector<double>> C = strassen(A, B);

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