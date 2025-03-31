import re
import pandas as pd
import spacy
from spacy import displacy
from spacy.matcher import Matcher
from tqdm import tqdm

nlp = spacy.load('en_core_web_sm')
pd.set_option('display.max_colwidth', 200)

# 加载数据
candidate_sentences = pd.read_csv("../data/wiki_sentences_v2.csv")

# 实体提取优化函数
def get_entities(sent):
    doc = nlp(sent)
    subj_entities = []
    obj_entities = []

    for token in doc:
        # 处理主语
        if token.dep_ in ['nsubj', 'nsubjpass']:
            for chunk in doc.noun_chunks:
                if chunk.root.i == token.i:
                    # 去除限定词
                    if chunk[0].dep_ == 'det' and len(chunk) > 1:
                        clean_chunk = chunk[1:].text
                    else:
                        clean_chunk = chunk.text
                    subj_entities.append(clean_chunk)
                    break
        
        # 处理宾语
        elif token.dep_ in ['dobj', 'pobj', 'iobj']:
            for chunk in doc.noun_chunks:
                if chunk.root.i == token.i:
                    if chunk[0].dep_ == 'det' and len(chunk) > 1:
                        clean_chunk = chunk[1:].text
                    else:
                        clean_chunk = chunk.text
                    obj_entities.append(clean_chunk)
                    break

    ent1 = subj_entities[0] if subj_entities else ""
    ent2 = obj_entities[0] if obj_entities else ""

    return [ent1.strip(), ent2.strip()]

# 关系提取优化函数
def get_relation(sent):
    doc = nlp(sent)
    root_verbs = [token for token in doc if token.dep_ == 'ROOT']
    if not root_verbs:
        return ""
    
    # 提取动词短语
    verb = root_verbs[0]
    phrase = []
    for child in verb.children:
        if child.dep_ in ('aux', 'auxpass', 'neg'):
            phrase.append(child.text)
    phrase.append(verb.lemma_)
    return ' '.join(phrase)

# 处理所有句子
entity_pairs = []
relations = []

for sent in tqdm(candidate_sentences["sentence"], desc="Processing sentences"):
    entities = get_entities(sent)
    entity_pairs.append(entities)
    relations.append(get_relation(sent))

# 创建数据框并清洗数据
kg_df = pd.DataFrame({
    'source': [pair[0] for pair in entity_pairs],
    'target': [pair[1] for pair in entity_pairs],
    'relation': relations
})

# 数据清洗步骤
# 1. 去除空值
kg_df = kg_df[(kg_df['source'] != '') & 
             (kg_df['target'] != '') & 
             (kg_df['relation'] != '')]

# 2. 过滤代词
pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
kg_df = kg_df[~kg_df['source'].str.lower().isin(pronouns) &
             ~kg_df['target'].str.lower().isin(pronouns)]

# 3. 标准化格式
kg_df['source'] = kg_df['source'].str.title()
kg_df['target'] = kg_df['target'].str.title()

# 4. 去除重复项
kg_df = kg_df.drop_duplicates(subset=['source', 'target', 'relation'])

# 保存结果
entities = pd.DataFrame(list(set(kg_df['source'].tolist() + kg_df['target'].tolist())),
                       columns=['entity'])
entities.to_csv('../results/entities.csv', index=False)
kg_df.to_csv('../results/relations.csv', index=False)