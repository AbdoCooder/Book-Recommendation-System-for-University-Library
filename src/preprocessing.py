from pathlib import Path
import pandas as pd


def _find_file(filename: str, base_dir: Path) -> Path:
    """Find a CSV file in data/raw first, then data/sample."""
    candidates = [
        base_dir / "data" / "raw" / filename,
        base_dir / "data" / "sample" / filename,
        base_dir / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {filename}. Put it in data/raw/ or data/sample/.")


def load_data(base_dir: str | Path = "."):
    base_dir = Path(base_dir)
    books_path = _find_file("Books.csv", base_dir)
    ratings_path = _find_file("Ratings.csv", base_dir)
    users_path = _find_file("Users.csv", base_dir)

    # Try utf-8 first, then latin-1 for Kaggle compatibility.
    def read(path):
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    books = read(books_path)
    ratings = read(ratings_path)
    users = read(users_path)
    return books, ratings, users


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    books = books.drop_duplicates(subset=["ISBN"])
    if "Year-Of-Publication" in books.columns:
        books["Year-Of-Publication"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
        books.loc[(books["Year-Of-Publication"] < 1900) | (books["Year-Of-Publication"] > 2026), "Year-Of-Publication"] = None
    return books


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    ratings = ratings.copy()
    ratings = ratings.drop_duplicates()
    ratings["Book-Rating"] = pd.to_numeric(ratings["Book-Rating"], errors="coerce")
    ratings = ratings.dropna(subset=["User-ID", "ISBN", "Book-Rating"])
    # Keep explicit ratings only. In the Kaggle dataset, 0 often means implicit/no rating.
    explicit = ratings[(ratings["Book-Rating"] >= 1) & (ratings["Book-Rating"] <= 10)]
    return explicit


def build_filtered_matrix(ratings: pd.DataFrame, min_user_ratings: int = 5, min_book_ratings: int = 5,
                          max_users: int = 3000, max_books: int = 3000) -> pd.DataFrame:
    """Create a manageable user-book matrix."""
    user_counts = ratings["User-ID"].value_counts()
    book_counts = ratings["ISBN"].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_books = book_counts[book_counts >= min_book_ratings].index

    filtered = ratings[ratings["User-ID"].isin(active_users) & ratings["ISBN"].isin(popular_books)]

    # In huge datasets, keep the most active users and most rated books.
    top_users = filtered["User-ID"].value_counts().head(max_users).index
    top_books = filtered["ISBN"].value_counts().head(max_books).index
    filtered = filtered[filtered["User-ID"].isin(top_users) & filtered["ISBN"].isin(top_books)]

    matrix = filtered.pivot_table(index="User-ID", columns="ISBN", values="Book-Rating", aggfunc="mean")
    return matrix