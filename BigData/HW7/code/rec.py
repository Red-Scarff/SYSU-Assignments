import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# 数据加载
ratings = pd.read_csv("../ml-latest-small/ratings.csv")
movies = pd.read_csv("../ml-latest-small/movies.csv")

# 数据预处理
movies["genres"] = movies["genres"].str.replace("|", " ")
train_data, test_data = train_test_split(ratings, test_size=0.2, stratify=ratings["userId"], random_state=42)

# 创建用户-电影矩阵
user_movie_train = train_data.pivot_table(index="userId", columns="movieId", values="rating", fill_value=0)
movie_ids = user_movie_train.columns.tolist()

# 基于内容的推荐特征工程
tfidf = TfidfVectorizer(stop_words="english")
genres_tfidf = tfidf.fit_transform(movies.set_index("movieId").loc[movie_ids]["genres"])
content_sim = cosine_similarity(genres_tfidf)
content_sim_df = pd.DataFrame(content_sim, index=movie_ids, columns=movie_ids)

# 用户协同过滤相似度计算
user_sim = cosine_similarity(user_movie_train)
user_sim_df = pd.DataFrame(user_sim, index=user_movie_train.index, columns=user_movie_train.index)

# 物品协同过滤相似度计算
item_sim = cosine_similarity(user_movie_train.T)
item_sim_df = pd.DataFrame(item_sim, index=movie_ids, columns=movie_ids)


def user_cf_recommend(user_id, n=10):
    if user_id not in user_movie_train.index:
        return []
    sim_users = user_sim_df[user_id].nlargest(11).index[1:]
    user_rated = user_movie_train.loc[user_id][user_movie_train.loc[user_id] > 0].index

    recommendations = {}
    for sim_user in sim_users:
        sim_ratings = user_movie_train.loc[sim_user]
        for movie_id in sim_ratings[sim_ratings > 0].index:
            if movie_id not in user_rated:
                recommendations[movie_id] = (
                    recommendations.get(movie_id, 0) + sim_ratings[movie_id] * user_sim_df.loc[user_id, sim_user]
                )
    return sorted(recommendations, key=recommendations.get, reverse=True)[:n]


def item_cf_recommend(user_id, n=10):
    if user_id not in user_movie_train.index:
        return []
    user_ratings = user_movie_train.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index

    scores = {}
    for movie_id in user_movie_train.columns:
        if user_ratings[movie_id] == 0:
            similar = item_sim_df[movie_id].nlargest(21).index[1:]
            total_sim = sum_score = 0
            for sm in similar:
                if sm in rated_movies:
                    sim = item_sim_df.loc[movie_id, sm]
                    total_sim += sim
                    sum_score += sim * user_ratings[sm]
            if total_sim > 0:
                scores[movie_id] = sum_score / total_sim
    return sorted(scores, key=scores.get, reverse=True)[:n]


def content_recommend(user_id, n=10):
    if user_id not in user_movie_train.index:
        return []
    user_ratings = user_movie_train.loc[user_id]
    liked = user_ratings[user_ratings >= 4].index

    candidates = {}
    for movie_id in liked:
        if movie_id not in content_sim_df.index:
            continue
        for similar in content_sim_df[movie_id].nlargest(21).index[1:]:
            if user_ratings[similar] == 0:
                candidates[similar] = candidates.get(similar, 0) + content_sim_df.loc[movie_id, similar]
    return sorted(candidates, key=candidates.get, reverse=True)[:n]


# 评估函数
def evaluate(model_func, threshold=4):
    test_users = test_data.userId.unique()
    total_hits = 0
    total_relevant = 0

    for user_id in tqdm(test_users):
        if user_id not in user_movie_train.index:
            continue
        relevant = test_data[(test_data.userId == user_id) & (test_data.rating >= threshold)].movieId
        if len(relevant) == 0:
            continue
        recommendations = model_func(user_id)
        hits = len(set(recommendations) & set(relevant))
        total_hits += hits
        total_relevant += len(relevant)

    return total_hits / total_relevant if total_relevant > 0 else 0


# 混合推荐
def hybrid_recommend(user_id, n=10):
    cf = {m: 0.7 for m in user_cf_recommend(user_id, 15)}
    cnt = {m: 0.3 for m in content_recommend(user_id, 15)}
    combined = {**cf, **cnt}
    for m in cnt:
        combined[m] += 0.3
    return sorted(combined, key=combined.get, reverse=True)[:n]


# 评估各模型
print("Evaluating User CF...")
user_cf_recall = evaluate(user_cf_recommend)
print("Evaluating Item CF...")
item_cf_recall = evaluate(item_cf_recommend)
print("Evaluating Content-Based...")
content_recall = evaluate(content_recommend)
print("Evaluating Hybrid...")
hybrid_recall = evaluate(hybrid_recommend)

# 结果展示
print(f"\nRecall Results:")
print(f"User Collaborative Filtering: {user_cf_recall:.4f}")
print(f"Item Collaborative Filtering: {item_cf_recall:.4f}")
print(f"Content-Based Recommendation: {content_recall:.4f}")
print(f"Hybrid Recommendation: {hybrid_recall:.4f}")")