import logging
import time
import warnings
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)


def extract_base_url(url: str) -> str:
    """
    Extracts the base URL (scheme + netloc) from a URl and converts it to lowercase.

    Parameters:
        url (str): URL to extract the base URL from.

    Returns:
        str: Base URL in lowercase.
    """
    url = url.lower()
    parsed_url = urlparse(url)
    base_url = f"{parsed_url.netloc}{parsed_url.path}"
    if parsed_url.params:
        base_url += f";{parsed_url.params}"

    if parsed_url.query:
        base_url += f"?{parsed_url.query}"

    return base_url


def find_url_duplicates(url_series: pd.Series, indexes_to_check: pd.Series | list[str] = None) -> pd.Series:
    """
    Finds duplicate URLs in a DataFrame.


    Parameters:
        url_series (pd.Series): Series containing the URLs, note that the index of the Series must be the entry number.
        indexes_to_check ([str] | None) (optional): Entry numbers to check, defaults to None (check all).
            Warning: we do not check urls that are not in the indexes_to_check, but we do bidirectionally list
            the duplicates. This means that an index that is in the `indexes_to_check` can be a duplicate to a
            index that is not in this list. Both of these indexes will have each other in their duplicates list.

    Returns:
        pd.Series: Series indicating duplicates by a list of indexes. Rows without duplicates are not returned.
    """
    # Verify input, check that the index is set
    if url_series.index.name is None:
        raise ValueError("The index of the DataFrame is not set, it is required to tag duplicates.")

    # Warn if the url_series is empty
    if url_series.empty:
        warning_text = "The url_series is empty, no duplicates will be found."
        warnings.warn(warning_text, UserWarning, stacklevel=1)
        logger.warning(warning_text)

    # Save the length of the original url_series for later logging
    original_url_series_length = len(url_series)

    start_time = time.time()

    # Copy the url_series to prevent modifying the original Series
    url_series = url_series.copy()

    # Drop all nan rows, log the number of dropped rows
    num_nan_rows = len(url_series[url_series.isna()])
    if num_nan_rows > 0:
        logger.info(f"Dropping {num_nan_rows}/{len(url_series)} rows with NaN values.")

    url_series = url_series.dropna()

    # Convert the URLs to the base URL
    url_series = url_series.apply(extract_base_url)

    # Reset index to work with both URLs and their original indices
    url_df = url_series.reset_index()
    url_dup_indexes = url_df.groupby("url").agg(list)["index"]

    # Drop all url_dup_indexes that only have one index, i.e. len == 1
    url_dup_indexes = url_dup_indexes[url_dup_indexes.apply(len) > 1]

    # Get a list of all urls that are in the indexes to check
    if indexes_to_check is not None:
        logger.info(
            f"Checking {len(indexes_to_check)} urls for duplicates. Warning: we do not check urls that are"
            "not in the indexes_to_check."
        )

        urls_to_check = url_series[url_series.index.isin(indexes_to_check)].unique()

        # Only select the url_dup_indexes that are in the urls_to_check if they exist
        urls_to_check = list(set(url_dup_indexes.index).intersection(set(urls_to_check)))
        url_dup_indexes = url_dup_indexes[urls_to_check]

    def set_duplicates_row(row: pd.Series) -> list[str] | type(pd.NA):  # type: ignore
        if row["url"] in url_dup_indexes:
            duplicate_indexes = url_dup_indexes[row["url"]].copy()
            # Remove its own index from the list of duplicates
            duplicate_indexes.remove(row["index"])

            return duplicate_indexes
        else:
            return pd.NA

    url_df["url_duplicates"] = url_df.apply(set_duplicates_row, axis=1)

    # Drop all rows that do not have duplicates
    url_df = url_df.dropna(subset=["url_duplicates"])

    # Create the final output Series, set the type to object explicitly to have consistent output when there
    # are no duplicates
    url_dup_series = url_df.set_index("index")["url_duplicates"].astype(object)

    end_time = time.time()
    logger.info(f"Time to find url duplicates: {round(end_time - start_time, 2)} seconds")

    if indexes_to_check is not None:
        n_dups = len(url_dup_series[url_dup_series.index.isin(indexes_to_check)])
        logger.info(f"Found {n_dups}/{len(indexes_to_check)} rows with duplicates in indexes_to_check")
    else:
        n_dups = len(url_dup_series)
        logger.info(f"Found {n_dups}/{original_url_series_length} rows with duplicates")

    return url_dup_series
