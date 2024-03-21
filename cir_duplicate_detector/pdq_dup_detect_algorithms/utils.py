import logging
import time
import warnings
from collections.abc import Callable
from multiprocessing import Pool, cpu_count
from typing import Any

import pandas as pd
import rapidfuzz

# Initialize logger
logger = logging.getLogger(__name__)

# Global constants
PDQ_HASH_LENGTH = 256


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


def hamming_distance(item1: str, item2: str) -> int:
    """
    Computes the Hamming distance between two PDQ hash items.
    Using the rapidfuzz library for speed.

    Parameters:
        item1 (str): First PDQ hash item.
        item2 (str): Second PDQ hash item.

    Returns:
        int: Hamming distance between item1 and item2.
    """

    # It seems unintuitive that the Hamming distance is computed on the binary strings
    # However, after trying to use the hamming distance on interegers, numpy arrays and bytes
    # using python hammingsdistance, numpy hammingsdistance and pybktree hammingdistance
    # it seems that rapidfuzz is orders of magnitude faster than the other methods
    return rapidfuzz.distance.Hamming.distance(item1, item2)


def run_in_parallel(
    worker_func: Callable[..., Any], tasks: tuple | Any, num_workers: int | None = None, chunk_size: int = 100
) -> list:
    """Runs tasks in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = cpu_count()
    start_time = time.time()

    def worker_wrapper(task: tuple | Any) -> Any:
        """
        This function is a wrapper for the worker function.
        It checks if the task is a tuple. If it is, it unpacks the tuple and passes the elements as arguments
        to the worker function. If the task is not a tuple, it passes the task directly to the worker function.
        """
        if isinstance(task, tuple):
            return worker_func(*task)
        else:
            return worker_func(task)

    if len(tasks) < chunk_size // 5:
        logger.info("Running single-threaded due to a small number of tasks.")
        results = [worker_wrapper(task) for task in tasks]
    else:
        logger.info(f"Running multi-threaded on {num_workers} cores.")

        with Pool(processes=num_workers) as pool:
            results = list(pool.imap(worker_wrapper, tasks, chunksize=chunk_size))
    end_time = time.time()
    logger.info(f"Time taken for parallel execution: {round(end_time - start_time, 2)} seconds")
    return results


def validate_similarity_threshold(threshold: float) -> None:
    """Validates the similarity threshold value."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")


def check_series_empty(series: pd.Series) -> bool:
    """Checks if the series is empty or has a single item and logs/warns accordingly."""
    if len(series) == 0:
        warnings.warn("The series is empty, no duplicates will be found.", UserWarning, stacklevel=1)
        logger.warning("The series is empty, no duplicates will be found.")
        return True
    return False
