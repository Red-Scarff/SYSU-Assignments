#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

// 打印矩阵的前 sub×sub 子块
void print_submatrix(double* M, int rows, int cols, int sub = 5) {
    for(int i = 0; i < min(sub, rows); ++i) {
        for(int j = 0; j < min(sub, cols); ++j)
            cout << M[i*cols + j] << "\t";
        if (cols > sub) cout << "...";
        cout << "\n";
    }
    if (rows > sub) cout << "...\n";
}

int main(int argc, char* argv[]) {
    if (argc < 6 || argc > 7) {
        cerr << "Usage: " << argv[0]
             << " m n k threads schedule [chunk]" << endl
             << " schedule: default|static|dynamic|guided" << endl
             << " chunk: positive integer (only for dynamic/guided)" << endl;
        return 1;
    }

    int m       = atoi(argv[1]);
    int n       = atoi(argv[2]);
    int k       = atoi(argv[3]);
    int threads = atoi(argv[4]);
    string sched= argv[5];
    int chunk   = 0;
    if ((sched == "dynamic" || sched == "guided")) {
        if (argc != 7) {
            cerr << "Error: chunk size must be specified for " << sched << " schedule." << endl;
            return 1;
        }
        chunk = atoi(argv[6]);
        if (chunk <= 0) {
            cerr << "Error: chunk size must be a positive integer." << endl;
            return 1;
        }
    }

    omp_set_num_threads(threads);  // 设置线程数

    // 固定随机种子
    srand(42);

    // 分配并初始化矩阵
    double* A = new double[m*n];
    double* B = new double[n*k];
    double* C = new double[m*k]{};

    for (int i = 0; i < m*n; ++i) A[i] = rand() % 100;
    for (int i = 0; i < n*k; ++i) B[i] = rand() % 100;

    // 验证输出
    cout << "Matrix A (first 2x2):\n";
    print_submatrix(A, m, n, 2);
    cout << "\nMatrix B (first 2x2):\n";
    print_submatrix(B, n, k, 2);

    // 计时并行乘法
    auto start = chrono::high_resolution_clock::now();

    if (sched == "default" || sched == "static") {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j) {
                double sum = 0;
                for (int l = 0; l < n; ++l)
                    sum += A[i*n + l] * B[l*k + j];
                C[i*k + j] = sum;
            }
    }
    else if (sched == "dynamic") {
        #pragma omp parallel for schedule(dynamic, chunk)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j) {
                double sum = 0;
                for (int l = 0; l < n; ++l)
                    sum += A[i*n + l] * B[l*k + j];
                C[i*k + j] = sum;
            }
    }
    else if (sched == "guided") {
        #pragma omp parallel for schedule(guided, chunk)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j) {
                double sum = 0;
                for (int l = 0; l < n; ++l)
                    sum += A[i*n + l] * B[l*k + j];
                C[i*k + j] = sum;
            }
    }
    else {
        cerr << "Unknown schedule: " << sched << "\n";
        return 1;
    }

    auto end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(end - start).count();

    // 结果输出
    cout << "\nMatrix C (first 2x2):\n";
    print_submatrix(C, m, k, 2);
    cout << "\nThreads: " << threads
         << ", Schedule: " << sched;
    if (sched == "dynamic" || sched == "guided") cout << ", Chunk: " << chunk;
    cout << ", Time: " << elapsed << " s\n";

    delete[] A; delete[] B; delete[] C;
    return 0;
}
