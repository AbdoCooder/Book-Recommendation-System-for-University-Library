from pathlib import Path
from typing import Any, Callable

import pandas as pd


CSV_READ_KWARGS: dict[str, Any] = {
    'sep': ';',
    'encoding': 'latin-1',
    'on_bad_lines': 'warn',
    'low_memory': False,
}


def _normalize_isbn(series: pd.Series) -> pd.Series:
    """Normalize ISBN-like values into a nullable string series.

    The function preserves the original index and does not drop rows. Each value
    is converted to pandas' nullable string dtype, leading and trailing whitespace
    is removed, and empty strings are converted to ``pd.NA`` so that downstream
    cleanup can treat blank identifiers as missing data.

    Args:
        series: A raw ISBN series that may contain strings, mixed types, or
            missing values.

    Returns:
        pd.Series: A nullable string series with whitespace stripped and blank
        values converted to ``pd.NA``.
    """
    return series.astype('string').str.strip().replace('', pd.NA)


def _coerce_required_int_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Drop invalid rows and cast a required identifier column to integers.

    This helper is used for columns such as ``User-ID`` where the values must be
    exact integers. Values are parsed with ``pd.to_numeric(errors='coerce')`` so
    malformed entries become missing. Rows with missing values are removed. Any
    remaining fractional values are rejected explicitly rather than being silently
    truncated by ``astype(int)``.

    Args:
        df: Input dataframe containing the target column.
        column_name: Name of the column that must contain integral values.

    Returns:
        pd.DataFrame: A copy of the dataframe containing only rows with valid
        integral values in ``column_name``, with that column cast to a non-null
        integer dtype.

    Raises:
        ValueError: If any non-missing values in ``column_name`` are fractional.
    """
    numeric_values = pd.to_numeric(df[column_name], errors='coerce')
    valid_values = numeric_values.dropna()

    if not valid_values.empty:
        non_integral_values = valid_values[valid_values.mod(1).ne(0)]
        if not non_integral_values.empty:
            raise ValueError(f"Column '{column_name}' contains non-integral values: {non_integral_values.tolist()}")

    cleaned_df = df.loc[numeric_values.notna()].copy()
    cleaned_df[column_name] = numeric_values.loc[cleaned_df.index].astype(int)
    return cleaned_df


def _coerce_required_rating_column(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and validate the ratings column used by the recommender.

    The raw file may use either ``Rating`` or the legacy ``Book-Rating`` column
    name. This helper selects whichever name is present, coerces the values to
    numeric, drops missing rows, and rejects fractional ratings explicitly. The
    cleaned dataframe always exposes a canonical ``Rating`` column typed as
    nullable integer ``Int64`` so downstream code can rely on a stable schema.

    Args:
        df: Input dataframe containing a ratings column.

    Returns:
        pd.DataFrame: A copy of the dataframe containing only rows with valid
        integral ratings, with the ratings column normalized to ``Rating`` and
        cast to ``Int64``.

    Raises:
        ValueError: If any non-missing rating values are fractional.
    """
    rating_column_name = 'Book-Rating' if 'Book-Rating' in df.columns else 'Rating'
    numeric_values = pd.to_numeric(df[rating_column_name], errors='coerce')
    valid_values = numeric_values.dropna()

    if not valid_values.empty:
        non_integral_values = valid_values[valid_values.mod(1).ne(0)]
        if not non_integral_values.empty:
            raise ValueError(
                f"Column '{rating_column_name}' contains non-integral values: {non_integral_values.tolist()}"
            )

    cleaned_df = df.loc[numeric_values.notna()].copy()
    cleaned_df[rating_column_name] = numeric_values.loc[cleaned_df.index].astype('Int64')
    if rating_column_name == 'Book-Rating':
        cleaned_df = cleaned_df.rename(columns={'Book-Rating': 'Rating'})
    return cleaned_df


def get_data_profile(
    df: pd.DataFrame,
    name: str,
    print_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build a serializable profile for a dataframe and optionally print it.

    The function is intentionally side-effect free unless ``print_fn`` is
    supplied. That makes it suitable for notebooks, tests, and pipelines that
    want structured metadata without forcing stdout output. The returned profile
    uses a plain ``dict[str, float]`` for missing-value percentages so the result
    is straightforward to serialize and has a consistent shape for both empty and
    non-empty inputs.

    Args:
        df: Input dataframe to profile.
        name: Human-readable label for the dataframe.
        print_fn: Optional callable that accepts a single string. When provided,
            the function emits a formatted summary through this callable.

    Returns:
        dict[str, Any]: A structured profile containing the dataframe name, row
        count, column list, and missing-value percentages.
    """
    if df.empty:
        missing_percentages: dict[str, float] = {column: 0.0 for column in df.columns}
    else:
        missing_percentages = {
            str(column): float(percentage)
            for column, percentage in df.isna().mean().mul(100).round(2).items()
        }

    profile: dict[str, Any] = {
        'name': name,
        'row_count': len(df),
        'columns': list(df.columns),
        'missing_percentages': missing_percentages,
    }

    if print_fn is not None:
        print_fn(f"--- {name} Profile ---")
        print_fn(f"Row count: {profile['row_count']}")
        print_fn(f"Columns: {profile['columns']}")
        print_fn("Missing values (%):")
        if missing_percentages:
            for column, percentage in missing_percentages.items():
                print_fn(f"{column}: {percentage:.2f}%")
        else:
            print_fn("<no columns>")
        print_fn("")

    return profile


def load_data(data_path: str | Path = 'data/raw/') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and clean the raw book-recommendation CSV files.

    This is the repository's main data-ingestion routine. It reads the raw CSV
    files from ``data_path``, applies a deterministic cleanup pipeline, and
    returns the cleaned users, books, and ratings dataframes as a tuple.

    Expected columns:
    - Users.csv: User-ID, Age
    - Books.csv: ISBN, Title, Author, Year, Publisher
    - Ratings.csv: User-ID, ISBN, Rating (or the legacy alias Book-Rating)

    Transformations applied:
    - User-ID values are coerced to numeric, rows with missing values are dropped,
        and the remaining values are validated to be integral before casting.
    - Age and Year are coerced to nullable integer (Int64) so missing values are
        preserved without inventing a sentinel value.
    - Ratings are coerced to nullable integer (Int64), malformed or missing values
        are dropped, and the output is normalized to the canonical ``Rating``
        column name.
    - ISBN values are normalized as stripped strings, with blank values converted
        to ``pd.NA`` and dropped in both books and ratings.
    - Missing Title, Author, and Publisher values are filled with ``'Unknown'``.
    - Raw CSV files are read with shared parsing options and malformed lines are
        surfaced with warnings.

    Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Cleaned users, books,
            and ratings dataframes.

    Raises:
            ValueError: If required identifier or rating values contain fractional
            numbers that cannot be represented as exact integers.
    """
    data_path = Path(data_path)

    def read_csv(filename: str) -> pd.DataFrame:
            return pd.read_csv(data_path / filename, **CSV_READ_KWARGS)

    users = read_csv('Users.csv')
    books = read_csv('Books.csv')
    ratings = read_csv('Ratings.csv')

    users = _coerce_required_int_column(users, 'User-ID')
    users['Age'] = pd.to_numeric(users['Age'], errors='coerce').astype('Int64')

    books['ISBN'] = _normalize_isbn(books['ISBN'])
    books = books.dropna(subset=['ISBN']).copy()
    books['Year'] = pd.to_numeric(books['Year'], errors='coerce').astype('Int64')
    books['Title'] = books['Title'].fillna('Unknown')
    books['Author'] = books['Author'].fillna('Unknown')
    books['Publisher'] = books['Publisher'].fillna('Unknown')

    ratings = _coerce_required_int_column(ratings, 'User-ID')
    ratings = _coerce_required_rating_column(ratings)
    ratings['ISBN'] = _normalize_isbn(ratings['ISBN'])
    ratings = ratings.dropna(subset=['ISBN']).copy()

    return users, books, ratings


def main() -> None:
    """Run the loader as a script and print basic dataframe profiles.

    This entry point is used when ``src/load_data.py`` is executed directly. It
    loads the cleaned datasets and prints one human-readable profile per
    dataframe.
    """
    users, books, ratings = load_data()

    get_data_profile(users, 'Users', print_fn=print)
    get_data_profile(books, 'Books', print_fn=print)
    get_data_profile(ratings, 'Ratings', print_fn=print)


if __name__ == '__main__':
    main()
