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
    """Calculates the absolute threshold from PDQ hash length and PQD hash similarity threshold.

    Args:
        pdq_hash_length: The length of the PDQ hash.
        pqd_hash_similarity_threshold: The PQD hash similarity threshold.

    Returns:
        The calculated absolute threshold.
    """
    return int(round(pdq_hash_length * (1 - pqd_hash_similarity_threshold)))


def hex_to_binary(hex_string: str, length: int | None = None) -> str:
    """Converts a hexadecimal string to a binary string.

    Args:
        hex_string: The hexadecimal string to convert.
        length: Optional; The desired length of the binary string.

    Returns:
        The binary string representation of the hexadecimal string.
    """
    if hex_string == "":
        logger.debug("Hex string is empty, returning empty string.")
        return ""

    integer = int(hex_string, 16)
    binary_string = format(integer, "b")

    if length is not None:
        binary_string = binary_string.zfill(length)

    return binary_string


def drop_literal_series_duplicates(series: pd.Series) -> pd.Series:
    """Drops all literal duplicate (index and value) rows from a series.

    Args:
        series: The series to drop duplicates from.

    Returns:
        A series with all literal duplicates removed.
    """
    series = series.copy()
    if series.empty:
        return series

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
        df.index.name = None

    return df[series_name]


def hamming_distance(item1: str, item2: str) -> int:
    """Computes the Hamming distance between two PDQ hash items.

    It seems unintuitive that the Hamming distance is computed on the binary strings. However, after trying to
    use the hamming distance on interegers, numpy arrays and bytes using python hammingsdistance, numpy
    hammingsdistance and pybktree hammingdistance it seems that rapidfuzz is orders of magnitude faster than
    the other methods.

    Args:
        item1: First PDQ hash item.
        item2: Second PDQ hash item.

    Returns:
        Hamming distance between item1 and item2.
    """
    return rapidfuzz.distance.Hamming.distance(item1, item2)


def run_in_parallel(
    worker_func: Callable[..., Any], tasks: tuple | Any, num_workers: int | None = None, chunk_size: int = 100
) -> list:
    """Runs tasks in parallel using multiprocessing.

    Args:
        worker_func: The function to run in parallel.
        tasks: The tasks to execute.
        num_workers: Optional; The number of worker processes.
        chunk_size: The number of tasks to process at a time.

    Returns:
        A list of results from the executed tasks.
    """
    if num_workers is None:
        num_workers = cpu_count()
    start_time = time.time()

    def worker_wrapper(task: tuple | Any) -> Any:
        """A wrapper to pass tasks to the worker function appropriately."""
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
    """Validates the similarity threshold value.
    Args:
        threshold: The similarity threshold to validate.

    Raises:
        ValueError: If the threshold is not between 0.0 and 1.0.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")


def check_series_empty(series: pd.Series) -> bool:
    """Checks if the series is empty or has a single item and logs/warns accordingly.

    Args:
        series: The series to check.

    Returns:
        True if the series is empty, False otherwise.
    """
    if len(series) == 0:
        warnings.warn("The series is empty, no duplicates will be found.", UserWarning, stacklevel=1)
        logger.warning("The series is empty, no duplicates will be found.")
        return True
    return False
