import logging
import time
import warnings

import pandas as pd

from cir_duplicate_detector.pdq_dup_detect_algorithms import naive_duplicate_detector
from cir_duplicate_detector.pdq_dup_detect_algorithms.bk_tree import PDQHashTree
from cir_duplicate_detector.pdq_dup_detect_algorithms.utils import (
    PDQ_HASH_LENGTH,
    drop_literal_series_duplicates,
    hex_to_binary,
)

logger = logging.getLogger(__name__)


def pdq_hash_output_formatter(
    pdq_hash_series: pd.Series, duplicate_detection_results: list[list[dict[str, str | int]]]
) -> pd.DataFrame:
    """Formats the duplicate detection results into a DataFrame.

    Args:
        pdq_hash_series: Series containing PDQ hash values.
        duplicate_detection_results: Nested list of dictionaries containing the results from the duplicate detection.

    Returns:
        DataFrame with `pdq_hash_duplicates` and `pdq_hash_similarities` columns.
    """
    # Format the duplicate_detection_results into a DataFrame of the desired output structure
    pdq_hash_dup_df = pd.DataFrame(index=pdq_hash_series.index.drop_duplicates())
    pdq_hash_dup_df["pdq_hash_duplicates"] = pd.NA
    pdq_hash_dup_df["pdq_hash_duplicates"] = pdq_hash_dup_df["pdq_hash_duplicates"].astype(object)
    pdq_hash_dup_df["pdq_hash_similarities"] = pd.NA
    pdq_hash_dup_df["pdq_hash_similarities"] = pdq_hash_dup_df["pdq_hash_similarities"].astype(object)

    for index, result in zip(pdq_hash_series.index, duplicate_detection_results, strict=True):
        # Format the result into a list of indexes, if no duplicates are found, set the value to pd.NA
        if len(result) > 0:
            # Since duplicate indexes are possible, we need to check if there is already a list;
            # if so, append to it. Since duplicates are bi-directional, we need to check if the index is
            # already in the list.
            for item in result:
                # Normalize the distance back to a percentage
                dist_normalized = 1.0 - (int(item["dist"]) / PDQ_HASH_LENGTH)

                # Update the duplicate of the source index
                if isinstance(pdq_hash_dup_df["pdq_hash_duplicates"][index], list):
                    if item["index"] not in pdq_hash_dup_df["pdq_hash_duplicates"][index] and index != item["index"]:
                        pdq_hash_dup_df.loc[index, "pdq_hash_duplicates"].append(item["index"])
                        pdq_hash_dup_df.loc[index, "pdq_hash_similarities"].append(dist_normalized)
                elif index != item["index"]:
                    pdq_hash_dup_df.loc[index, "pdq_hash_duplicates"] = [item["index"]]
                    pdq_hash_dup_df.loc[index, "pdq_hash_similarities"] = [dist_normalized]

                # Update the duplicate of the duplicate index
                if item["index"] not in pdq_hash_dup_df.index:
                    pdq_hash_dup_df.loc[item["index"]] = pd.NA
                if isinstance(pdq_hash_dup_df["pdq_hash_duplicates"][item["index"]], list):
                    if index not in pdq_hash_dup_df["pdq_hash_duplicates"][item["index"]] and index != item["index"]:
                        pdq_hash_dup_df.loc[item["index"], "pdq_hash_duplicates"].append(index)
                        pdq_hash_dup_df.loc[item["index"], "pdq_hash_similarities"].append(dist_normalized)
                elif index != item["index"]:
                    pdq_hash_dup_df.loc[item["index"], "pdq_hash_duplicates"] = [index]
                    pdq_hash_dup_df.loc[item["index"], "pdq_hash_similarities"] = [dist_normalized]

    # Drop all rows that do not have duplicates
    pdq_hash_dup_df = pdq_hash_dup_df.dropna()

    # Sort the indexes
    pdq_hash_dup_df = pdq_hash_dup_df.sort_index()

    return pdq_hash_dup_df


def find_pdq_hash_duplicates(
    pdq_hash_series: pd.Series,
    indexes_to_check: (list[str] | None) = None,
    pqd_hash_similarity_threshold: float = 0.2,
    duplicate_detection_method: str = "bk-tree",
) -> pd.DataFrame:
    """Finds perceptual hash duplicates in a DataFrame. The determination of duplicates is based on the
    hamming distance > hash_similarity_threshold.

    Overview of the function:
    - Explode the pdq_hash column to get a DataFrame with one hash per row. This results in a DataFrame
      with duplicate indexes with different hashes.
    - Filter the DataFrame to only contain the indexes to check.
    - Use either the naive method or the bk-tree method to find duplicates.
    - If the naive method is used, each row is compared to all other rows in the DataFrame.
    - If the bk-tree method is used, the following steps are taken:
        - Build a BK-tree from the DataFrame.
        - For each index, find the duplicates in the BK-tree.
    - If the multi index hashing method is used, the following steps are taken:
        #TODO:
    - Aggregate the duplicates back to unique indexes.
    - Only return the indexes that are in the indexes_to_check list and have duplicates.

    Args:
        pdq_hash_series: Series containing the perceptual hashes.
        indexes_to_check: Optional; Indexes to check, defaults to None.
        pqd_hash_similarity_threshold: Threshold for Hamming distance to determine hash similarity, defaults to 0.2.
        duplicate_detection_method: Method to use for duplicate detection, defaults to "bk-tree".

    Returns:
        DataFrame containing duplicates and their similarities.
    """
    # Verify input, ensure the index is set
    if pdq_hash_series.index.name is None:
        raise ValueError("The index of the DataFrame is not set, it is required to tag duplicates.")

    # Verify the pdq_hash_similarity_threshold is valid
    if not isinstance(pqd_hash_similarity_threshold, float):
        raise ValueError(f"pqd_hash_similarity_threshold must be a float, got {type(pqd_hash_similarity_threshold)}")

    if pqd_hash_similarity_threshold < 0.0 or pqd_hash_similarity_threshold > 1.0:
        raise ValueError(
            f"pqd_hash_similarity_threshold must be between 0.0 and 1.0, got {pqd_hash_similarity_threshold}"
        )

    # Verify pdq_hash_series is a Series
    if not isinstance(pdq_hash_series, pd.Series):
        raise ValueError(f"pdq_hash_series must be a pandas Series, got {type(pdq_hash_series)}")

    def empty_output_df() -> pd.DataFrame:
        """Create an empty output dataframe with the correct index name and dtype."""
        empty_df = pd.DataFrame(columns=["pdq_hash_duplicates", "pdq_hash_similarities"])
        empty_df["pdq_hash_duplicates"] = empty_df["pdq_hash_duplicates"].astype(object)
        empty_df["pdq_hash_similarities"] = empty_df["pdq_hash_similarities"].astype(object)
        empty_df.index.name = pdq_hash_series.index.name
        empty_df.index = empty_df.index.astype(pdq_hash_series.index.dtype)
        return empty_df

    # Warn if the pdq_hash_series is empty
    if pdq_hash_series.empty:
        warnings.warn("The pdq_hash_series is empty, no duplicates will be found.", UserWarning, stacklevel=1)
        logger.warning("The pdq_hash_series is empty, no duplicates will be found.")
        return empty_output_df()

    # If indexes_to_check is empty
    if indexes_to_check is not None and len(indexes_to_check) == 0:
        warnings.warn("indexes_to_check is empty, no duplicates will be found.", UserWarning, stacklevel=1)
        logger.warning("indexes_to_check is empty, no duplicates will be found.")
        return empty_output_df()

    # Save the length of the original pdq_hash_series for later logging
    original_pdq_hash_series_length = len(pdq_hash_series)

    logger.info(f"Using similarity threshold: {pqd_hash_similarity_threshold}.")

    start_time = time.time()

    # Copy the pdq_hash_series to prevent modifying the original Series
    pdq_hash_series = pdq_hash_series.copy()

    # Drop all NaN rows, log the number of dropped rows
    num_nan_rows = pdq_hash_series.isna().sum()
    if num_nan_rows > 0:
        logger.info(f"Dropping {num_nan_rows} rows with NaN values out of {len(pdq_hash_series)}")
        pdq_hash_series = pdq_hash_series.dropna()

    # Explode the pdq_hash column to get a DataFrame with one hash per row
    pdq_hash_series = pdq_hash_series.explode()

    # Verify that all rows are of type string, None, or pd.NA
    # Filter out all rows that are NaN
    non_string_rows = pdq_hash_series[~pdq_hash_series.isna()].map(type) != str
    if non_string_rows.any():
        rows_with_non_string_values = pdq_hash_series[non_string_rows]
        raise ValueError(
            f"pdq_hash_series must be a pandas Series of type string, got {non_string_rows.unique()}\n"
            f"The following rows contain non-string values: {rows_with_non_string_values}"
        )

    # Transform the pdq_hash_series to a Series of type string
    pdq_hash_series = pdq_hash_series.astype(pd.StringDtype())

    # Drop all literal duplicate (index and value) rows, log the number of dropped rows
    pdq_hash_series = drop_literal_series_duplicates(pdq_hash_series)

    # Convert the hex strings to binary strings
    # The PDQ hash is 256 bits, therefore we need to pad the binary string to 256 bits
    # The padding is needed since leading zeros could have been removed in the hex format
    # It seems unintuitive that the Hamming distance is computed on the binary strings
    # However, after trying to use the hamming distance on integers, numpy arrays, and bytes
    # using python hammingsdistance, numpy hammingsdistance, and pybktree hammingdistance,
    # it seems that rapidfuzz is orders of magnitude faster than the other methods
    pdq_hash_series = pdq_hash_series.apply(hex_to_binary, length=PDQ_HASH_LENGTH)

    # There are three methods for finding duplicates: naive, bk-tree, and mih
    # The bk-tree method was implemented first, but the performance over the naive method was minimal
    # The naive method is simpler and easier to understand. It is also faster for small datasets,
    # therefore, it is the default method
    # TODO: update this comment based on any new developments or insights
    if duplicate_detection_method.lower() == "naive":
        logger.info("Using the naive method for duplicate detection.")
        duplicate_detection_results = naive_duplicate_detector.get_pdq_fuzzy_duplicates(
            pdq_hash_series=pdq_hash_series,
            pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
            indexes_to_check=indexes_to_check,
        )
    elif duplicate_detection_method.lower() == "bk-tree":
        logger.info("Using the BK-tree method for duplicate detection.")

        # Build the BK-tree
        pdq_hash_tree = PDQHashTree(pdq_hash_series)

        # Filter the DataFrame to only contain the indexes to check, when None all entries are checked
        if indexes_to_check is not None:
            # Note that the .dropna() can drop indexes that are in the indexes_to_check list
            # Therefore we can only select indexes if they exists
            pdq_hash_series = pdq_hash_series[pdq_hash_series.index.isin(indexes_to_check)]

        # Find the duplicates
        duplicate_detection_results = pdq_hash_tree.get_duplicates(
            pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold=pqd_hash_similarity_threshold
        )
    elif duplicate_detection_method.lower() == "mih":
        # TODO: Implement the multi-index hashing method for efficient large-scale duplicate detection
        raise NotImplementedError("The Multi-Index Hashing (MIH) method is not implemented yet.")
    else:
        raise ValueError(f"Unknown duplicate detection method: {duplicate_detection_method}")

    # Reduce pdq_hash_series to only the indexes to check
    # Filter the DataFrame to only contain the indexes to check, when None all entries are checked
    if indexes_to_check is not None:
        # Note that the .dropna() can drop indexes that are in the indexes_to_check list
        # Therefore we can only select indexes if they exists
        pdq_hash_series = pdq_hash_series[pdq_hash_series.index.isin(indexes_to_check)]

    # Format the duplicate detection results into a DataFrame with the desired output structure
    pdq_hash_duplicates_df = pdq_hash_output_formatter(
        pdq_hash_series=pdq_hash_series, duplicate_detection_results=duplicate_detection_results
    )

    end_time = time.time()
    if indexes_to_check is not None:
        logger.info(
            f"Time taken to check {len(indexes_to_check)} rows for duplicates: "
            f"{round(end_time - start_time, 2)} seconds"
        )

        indexes_to_check_with_duplicates_length = len(
            pdq_hash_duplicates_df[pdq_hash_duplicates_df.index.isin(indexes_to_check)]
        )
        logger.info(
            f"Found {indexes_to_check_with_duplicates_length}/{len(indexes_to_check)} rows in "
            "`indexes_to_check` with duplicates."
        )
    else:
        logger.info(
            f"Time taken to check {original_pdq_hash_series_length} rows for duplicates: "
            f"{round(end_time - start_time, 2)} seconds"
        )
        logger.info(f"Found {len(pdq_hash_duplicates_df)}/{original_pdq_hash_series_length} rows with duplicates.")

    return pdq_hash_duplicates_df
