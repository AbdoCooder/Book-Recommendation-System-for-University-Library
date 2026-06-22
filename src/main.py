from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_data, clean_books, clean_ratings, build_filtered_matrix
from recommender import item_similarity_matrix, recommend_for_user
from clustering import cluster_users
from evaluation import evaluate_baseline_rmse

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = BASE_DIR / "outputs"
(OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUTS / "tables").mkdir(parents=True, exist_ok=True)

books, ratings, users = load_data(BASE_DIR)
books = clean_books(books)
ratings = clean_ratings(ratings)

print("Books:", books.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)

# EDA
ratings["Book-Rating"].hist()
plt.title("Distribution des ratings")
plt.xlabel("Rating")
plt.ylabel("Fréquence")
plt.savefig(OUTPUTS / "figures" / "ratings_distribution.png", bbox_inches="tight")
plt.close()

matrix = build_filtered_matrix(ratings)
print("User-book matrix:", matrix.shape)

# Recommender
if len(matrix.index) > 0:
    example_user = matrix.index[0]
    recs = recommend_for_user(example_user, matrix, books, top_n=10)
    recs.to_csv(OUTPUTS / "tables" / "recommendations_example.csv", index=False)
    print(recs)

# Evaluation
kpis = evaluate_baseline_rmse(ratings)
pd.DataFrame([kpis]).to_csv(OUTPUTS / "tables" / "evaluation_rmse.csv", index=False)
print(kpis)

# Clustering
clusters, score = cluster_users(matrix, n_clusters=4)
clusters.to_csv(OUTPUTS / "tables" / "user_clusters.csv", index=False)
print("Silhouette score:", score)
print(clusters["Cluster"].value_counts())