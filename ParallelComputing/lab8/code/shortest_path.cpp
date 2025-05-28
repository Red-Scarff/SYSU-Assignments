#include <iostream>
#include <fstream>
#include <vector>
#include <climits>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const double INF = 1e9;

class Graph {
private:
    int n;
    vector<vector<double>> dist;
    
public:
    Graph(int vertices) : n(vertices) {
        dist.assign(n, vector<double>(n, INF));
        for (int i = 0; i < n; i++) {
            dist[i][i] = 0.0;
        }
    }
    
    void addEdge(int u, int v, double w) {
        dist[u][v] = min(dist[u][v], w);
        dist[v][u] = min(dist[v][u], w);
    }
    
    // 优化的并行Floyd-Warshall
    void floydWarshallOptimized() {
        for (int k = 0; k < n; k++) {
            // 只并行化外层循环，减少内存竞争
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) {
                if (dist[i][k] < INF) {  // 提前检查避免无效计算
                    for (int j = 0; j < n; j++) {
                        if (dist[k][j] < INF) {
                            double new_dist = dist[i][k] + dist[k][j];
                            if (new_dist < dist[i][j]) {
                                dist[i][j] = new_dist;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 分块并行版本（适合大图）
    void floydWarshallBlocked(int block_size = 64) {
        for (int k = 0; k < n; k += block_size) {
            int k_end = min(k + block_size, n);
            
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < n; i += block_size) {
                int i_end = min(i + block_size, n);
                
                for (int j = 0; j < n; j += block_size) {
                    int j_end = min(j + block_size, n);
                    
                    // 块内计算
                    for (int kk = k; kk < k_end; kk++) {
                        for (int ii = i; ii < i_end; ii++) {
                            if (dist[ii][kk] < INF) {
                                for (int jj = j; jj < j_end; jj++) {
                                    if (dist[kk][jj] < INF) {
                                        double new_dist = dist[ii][kk] + dist[kk][jj];
                                        if (new_dist < dist[ii][jj]) {
                                            dist[ii][jj] = new_dist;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    double getDistance(int u, int v) {
        return dist[u][v];
    }
    
    int getVertexCount() {
        return n;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <adjacency_file> <test_file> [num_threads] [algorithm]" << endl;
        cout << "Algorithm: 0=optimized (default), 1=blocked" << endl;
        return 1;
    }
    
    string adj_file = argv[1];
    string test_file = argv[2];
    int num_threads = (argc >= 4) ? atoi(argv[3]) : omp_get_max_threads();
    int algorithm = (argc >= 5) ? atoi(argv[4]) : 0;
    
    omp_set_num_threads(num_threads);
    
    // 读取图数据
    ifstream adj_input(adj_file);
    if (!adj_input.is_open()) {
        cerr << "Error: Cannot open adjacency file " << adj_file << endl;
        return 1;
    }
    
    int max_vertex = 0;
    string line;
    getline(adj_input, line); // Skip header
    
    while (getline(adj_input, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string item;
        
        getline(ss, item, ',');
        int u = stoi(item);
        getline(ss, item, ',');
        int v = stoi(item);
        
        max_vertex = max(max_vertex, max(u, v));
    }
    adj_input.close();
    
    Graph graph(max_vertex + 1);
    
    // 添加边
    adj_input.open(adj_file);
    getline(adj_input, line); // Skip header
    
    while (getline(adj_input, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        string item;
        
        getline(ss, item, ',');
        int u = stoi(item);
        getline(ss, item, ',');
        int v = stoi(item);
        getline(ss, item, ',');
        double w = stod(item);
        
        graph.addEdge(u, v, w);
    }
    adj_input.close();
    
    cout << "Graph loaded: " << graph.getVertexCount() << " vertices" << endl;
    cout << "Threads: " << num_threads << ", Algorithm: " << 
            (algorithm == 0 ? "optimized" : "blocked") << endl;
    
    // 计算最短路径
    auto start_time = high_resolution_clock::now();
    
    if (algorithm == 0) {
        graph.floydWarshallOptimized();
    } else {
        graph.floydWarshallBlocked();
    }
    
    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    
    cout << "Time: " << duration.count() << " ms" << endl;
    
    // 处理测试查询
    ifstream test_input(test_file);
    if (!test_input.is_open()) {
        cerr << "Error: Cannot open test file " << test_file << endl;
        return 1;
    }
    
    cout << "\nResults:" << endl;
    while (getline(test_input, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        int u, v;
        ss >> u >> v;
        
        double distance = graph.getDistance(u, v);
        if (distance >= INF) {
            cout << u << " -> " << v << ": inf" << endl;
        } else {
            cout << u << " -> " << v << ": " << distance << endl;
        }
    }
    
    test_input.close();
    return 0;
}