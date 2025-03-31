from py2neo import Graph, Node, Subgraph
import pandas as pd
import re

# Neo4j连接配置
graph = Graph("bolt://localhost:7687", 
             auth=("neo4j", "W_zj204526922"))

# ===== 清除旧数据 =====
clear_existing = input("是否清除现有图谱数据？(y/n): ").lower() == 'y'
if clear_existing:
    graph.run("MATCH (n) DETACH DELETE n")
    print("已清除旧数据")
else:
    print("保留现有数据，继续追加")

# 创建唯一性约束
graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")

# ===== 实体处理 =====
entities = pd.read_csv("../results/entities.csv")
entities['entity'] = entities['entity'].astype(str).str.strip()
entities = entities[~entities['entity'].str.match(r'^\s*$')]

# 批量创建节点
nodes = [Node("Entity", name=row['entity']) for _, row in entities.iterrows()]
graph.create(Subgraph(nodes))

# ===== 关系处理 ===== 
relations = pd.read_csv("../results/relations.csv")

# 数据清洗增强版
relations = relations.dropna(subset=['source', 'target', 'relation'])
relations['source'] = relations['source'].str.strip()
relations['target'] = relations['target'].str.strip()

# 处理关系类型命名规范
def clean_relation_type(rel):
    # 替换非法字符为下划线，保留字母数字和下划线
    rel = re.sub(r'[^\w]', '_', str(rel))  
    # 合并连续下划线
    rel = re.sub(r'_+', '_', rel)          
    # 去除首尾下划线并大写
    rel = rel.strip('_').upper()            
    return rel if rel else 'RELATED_TO'     # 空值兜底

relations['relation'] = relations['relation'].apply(clean_relation_type)

# 按关系类型分组处理
grouped = relations.groupby('relation')

for rel_type, group in grouped:
    # 分批提交（每组关系类型单独处理）
    batch_size = 1000
    for i in range(0, len(group), batch_size):
        batch_data = group[i:i+batch_size].to_dict('records')
        
        # 动态生成Cypher查询
        query = f"""
        UNWIND $data AS row
        MATCH (a:Entity {{name: row.source}}), (b:Entity {{name: row.target}})
        MERGE (a)-[:`{rel_type}`]->(b)
        """
        graph.run(query, data=batch_data)

print(f"成功导入：{len(entities)} 个实体，{len(relations)} 条关系（共 {len(grouped)} 种关系类型）")