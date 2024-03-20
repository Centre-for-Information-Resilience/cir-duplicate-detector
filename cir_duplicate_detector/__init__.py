import logging

import pandas as pd

from .pdq_hash import find_pdq_hash_duplicates
from .url import find_url_duplicates

logger = logging.getLogger(__name__)


def detect_duplicates(
    df: pd.DataFrame,
    indexes_to_check: (list[str] | None) = None,
    pqd_hash_similarity_threshold: float = 0.8,
    pdq_duplicate_detection_method: str = "pandas",
) -> pd.DataFrame:
    """
    Detect duplicate entries based on URLs and perceptual hash similarity.

    Parameters:
    df (pd.DataFrame): Input DataFrame with the following columns:
        - index (str): Entry number
        - url (str) (optional): URL
        - pdq_hash (list(str)) (optional): Perceptual hash
        Note that either url or pdq_hash must be present.
    indexes_to_check ([str] | None) (optional): Entry numbers to check, defaults to None (check all).
    pqd_hash_similarity_threshold (float) (optional): Threshold (percentage) for Hamming distance to determine
        hash similarity. Where 1.0 is a perfect bit for bit match and 0.0 is no match at all. Defaults to 0.8.
    pdq_duplicate_detection_method (str) (optional): Method to use for detecting perceptual hash duplicates.
        Options: "pandas" or "bk-tree". Defaults to "pandas".

    Returns:
    pd.DataFrame: DataFrame with columns indicating duplicates.
    """
    # Validate input DataFrame
    if "url" not in df.columns and "pdq_hash" not in df.columns:
        raise ValueError(f"Column `url` or `pdq_hash` not found in dataframe. Found columns: {df.columns}")

    # Check if the index is set or an in index column exists
    if df.index.name is None and "index" not in df.columns:
        raise ValueError(
            "The DataFrame index is not set and no index column is found. Please set the index or add "
            "an 'index' column."
        )

    # Select only the necessary columns (if they exist) and copy these to prevent modifying the original DataFrame
    columns_to_copy = []

    if "url" in df.columns:
        columns_to_copy.append("url")

    if "pdq_hash" in df.columns:
        columns_to_copy.append("pdq_hash")

    if "index" in df.columns:
        columns_to_copy.append("index")

    # The index is always copied
    df = df[columns_to_copy].copy()

    # Check if an index column is set
    if "index" in df.columns:
        # Set the index to the index column
        df = df.set_index("index")
    elif df.index.name is None:
        logger.warning("No index column found and no custom index set. Using the DataFrame row numbers as index.")

    # Check that the indexes are unique
    non_unique_indexes = df.index[df.index.duplicated()].unique()

    if len(non_unique_indexes) > 0:
        raise ValueError(f"The DataFrame indexes are not unique. Non-unique indexes: {non_unique_indexes}")

    # Make sure the df is a dataframe
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Create an output df only containing the index of the df
    output_df = pd.DataFrame(index=df.index)

    # Find URL duplicates
    if "url" in df.columns:
        # Find URL duplicates, only pass the index and the relevant column
        url_duplicates = find_url_duplicates(url_series=df["url"], indexes_to_check=indexes_to_check)

        # Join the URL duplicates to the output DataFrame
        output_df = output_df.merge(url_duplicates, on="index", how="left")
    else:
        logger.warning("Column `url` not found in dataframe. Skipping URL duplicate check.")

    # Find perceptual hash duplicates
    if "pdq_hash" in df.columns:
        perceptual_hash_duplicates = find_pdq_hash_duplicates(
            pdq_hash_series=df["pdq_hash"],
            indexes_to_check=indexes_to_check,
            pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
            duplicate_detection_method=pdq_duplicate_detection_method,
        )

        # Join the perceptual hash duplicates to the output DataFrame
        output_df = output_df.merge(perceptual_hash_duplicates, on="index", how="left")
    else:
        logger.warning("Column `pdq_hash` not found in dataframe. Skipping perceptual hash duplicate check.")

    # Only return rows with any duplicates
    output_df = output_df.dropna(how="all")

    # Add the index as a column put it at as the first column
    output_df.insert(0, "index", output_df.index)

    return output_df
