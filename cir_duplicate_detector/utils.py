import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def hex_to_binary(hex_string: str, length: int | None = None) -> str:
    """
    Convert a hexadecimal string to a binary string.

    Args:
        hex_string (str): The hexadecimal string to convert.
        length (int, optional): The desired length of the binary string. Defaults to None.

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
    """
    Drop all literal duplicate (index and value) rows from a series.

    Parameters:
        series (pd.Series): Series to drop duplicates from.

    Returns:
        pd.Series: Series with all literal duplicates removed.
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


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return basic information about each column in a DataFrame.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - info_df: DataFrame containing information about each column
    """
    # Copy the DataFrame to prevent modifying the original
    df = df.copy()

    info_list = []
    total_rows = len(df)

    # Check if the index is set, if so reset it to make sure it is a column
    if df.index.name is not None:
        index_dtype = df.index.dtype
        index_name = df.index.name
        df = df.reset_index()
        df[index_name] = df[index_name].astype(index_dtype)

    for col in df.columns:
        # Check if a row is a list or a numpy.ndarray, if so convert it to a tuple to make it hashable
        if df[col].apply(lambda x: isinstance(x, list | np.ndarray)).any():
            df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list | np.ndarray) else x)

        # Basic column information
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        missing_count = df[col].isna().sum()
        missing_percent = (missing_count / total_rows) * 100
        missing_abs = f"{missing_count}/{total_rows}"

        # Most common value and its frequency
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else None
        mode_freq = (df[col] == mode_val).sum() if mode_val is not None else None

        # Append information to the list
        info_list.append([col, dtype, unique_count, missing_count, missing_abs, missing_percent, mode_val, mode_freq])

    # Convert list to DataFrame
    info_df = pd.DataFrame(
        info_list,
        columns=[
            "Column",
            "Data Type",
            "Unique Values",
            "Missing Values",
            "Missing (Absolute)",
            "Missing %",
            "Mode",
            "Mode Frequency",
        ],
    )

    return info_df
