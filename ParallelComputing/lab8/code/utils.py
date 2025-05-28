import pandas as pd


def analyze_graph(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 计算节点数量（唯一节点）
    all_nodes = pd.concat([df["source"], df["target"]])
    num_nodes = len(all_nodes.unique())

    # 计算边数量
    num_edges = len(df)

    # 计算平均度数（有向图）
    avg_degree = num_edges / num_nodes

    return num_nodes, num_edges, avg_degree


if __name__ == "__main__":
    file_path = "updated_flower.csv"  # 替换为实际文件路径
    nodes, edges, avg_deg = analyze_graph(file_path)

    print(f"节点数量 [V1]: {nodes}")
    print(f"边数量 [E1]: {edges}")
    print(f"平均度数 [D1]: {avg_deg:.2f}")
