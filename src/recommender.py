import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def item_similarity_matrix(user_book_matrix: pd.DataFrame) -> pd.DataFrame:
    filled = user_book_matrix.fillna(0)
    sim = cosine_similarity(filled.T)
    return pd.DataFrame(sim, index=filled.columns, columns=filled.columns)


def recommend_similar_books(book_isbn: str, sim_items: pd.DataFrame, books: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if book_isbn not in sim_items.index:
        return pd.DataFrame()
    scores = sim_items[book_isbn].sort_values(ascending=False).drop(book_isbn).head(top_n)
    result = scores.reset_index()
    result.columns = ["ISBN", "similarity"]
    return result.merge(books[["ISBN", "Book-Title", "Book-Author"]], on="ISBN", how="left")


def recommend_for_user(user_id, user_book_matrix: pd.DataFrame, books: pd.DataFrame, top_n: int = 10, n_neighbors: int = 20) -> pd.DataFrame:
    filled = user_book_matrix.fillna(0)
    if user_id not in filled.index:
        return pd.DataFrame()

    sim = cosine_similarity(filled)
    sim_df = pd.DataFrame(sim, index=filled.index, columns=filled.index)
    neighbors = sim_df[user_id].drop(user_id).sort_values(ascending=False).head(n_neighbors)

    neighbor_ratings = filled.loc[neighbors.index]
    weighted_scores = np.dot(neighbors.values, neighbor_ratings) / (np.abs(neighbors.values).sum() + 1e-9)
    scores = pd.Series(weighted_scores, index=filled.columns)

    already_read = filled.loc[user_id]
    scores = scores[already_read == 0].sort_values(ascending=False).head(top_n)

    result = scores.reset_index()
    result.columns = ["ISBN", "score"]
    return result.merge(books[["ISBN", "Book-Title", "Book-Author"]], on="ISBN", how="left")