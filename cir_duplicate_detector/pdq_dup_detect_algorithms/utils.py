import logging

import pandas as pd

logger = logging.getLogger(__name__)


def calculate_absolute_threshold(pdq_hash_length: int, pqd_hash_similarity_threshold: float) -> int:
    """Calculates the absolute threshold based on the PDQ hash length and the PQD hash similarity threshold.

    Args:
        pdq_hash_length (int): The length of the PDQ hash.
        pqd_hash_similarity_threshold (float): The PQD hash similarity threshold.

    Returns:
        int: The calculated absolute threshold.
    """
    return int(round(pdq_hash_length * (1 - pqd_hash_similarity_threshold)))


def hex_to_binary(hex_string: str, length: int | None = None) -> str:
    """Converts a hexadecimal string to a binary string.

    Args:
        hex_string (str): The hexadecimal string to convert.
        length (int | None, optional): The desired length of the binary string. Defaults to None.

    Returns:
        str: The binary string representation of the hexadecimal string.
    """
    if hex_string == "":
        logger.debug("Hex string is empty, returning empty string.")
        return ""

    # Convert hex string to an integer
    integer = int(hex_string, 16)

    # return integer

    # Convert the integer to a binary string
    binary_string = format(integer, "b")

    # If a specific length is required, pad with leading zeros
    if length is not None:
        binary_string = binary_string.zfill(length)

    return binary_string


def drop_literal_series_duplicates(series: pd.Series) -> pd.Series:
    """Drops all literal duplicate (index and value) rows from a series.

    Args:
        series (pd.Series): The series to drop duplicates from.

    Returns:
        pd.Series: A series with all literal duplicates removed.
    """

    # Copy the series to prevent modifying the original Series
    series = series.copy()

    # Check if the series is empty, if so return the empty series
    if series.empty:
        return series

    # Remove all literal duplicate (index and value) rows, log the number of dropped rows
    # Reset the index to be able to use the duplicated function
    index_name = series.index.name
    if index_name is None:
        logger.warning("Index name is not set, if no custom index is used, pd.dropna() preferred.")
    series_name = series.name

    df = series.reset_index()
    num_duplicate_rows = len(df[df.duplicated()])

    if num_duplicate_rows > 0:
        logger.info(f"Dropping {num_duplicate_rows}/{len(df)} rows with duplicate values.")

    df = df.drop_duplicates()

    if index_name is not None:
        df = df.set_index(index_name)
    else:
        df = df.set_index("index")
        # Drop the index name since it was not set before
        df.index.name = None

    return df[series_name]
