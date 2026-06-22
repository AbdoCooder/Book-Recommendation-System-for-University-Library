import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def evaluate_baseline_rmse(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    global_mean = train["Book-Rating"].mean()
    user_mean = train.groupby("User-ID")["Book-Rating"].mean()
    item_mean = train.groupby("ISBN")["Book-Rating"].mean()

    pred_global = np.full(len(test), global_mean)
    pred_user = test["User-ID"].map(user_mean).fillna(global_mean)
    pred_item = test["ISBN"].map(item_mean).fillna(global_mean)

    return {
        "RMSE global mean": float(np.sqrt(mean_squared_error(test["Book-Rating"], pred_global))),
        "RMSE user mean": float(np.sqrt(mean_squared_error(test["Book-Rating"], pred_user))),
        "RMSE item mean": float(np.sqrt(mean_squared_error(test["Book-Rating"], pred_item))),
    }


def simple_precision_at_k(recommendations: dict, relevant_items: dict, k: int = 10):
    hits = 0
    total_recommended = 0
    total_users = 0
    for user, recs in recommendations.items():
        if user not in relevant_items:
            continue
        top_recs = set(recs[:k])
        relevant = set(relevant_items[user])
        hits += len(top_recs & relevant)
        total_recommended += k
        total_users += 1
    precision = hits / total_recommended if total_recommended else 0
    hit_rate = hits / total_users if total_users else 0
    return {"Precision@K": precision, "HitRate@K": hit_rate}