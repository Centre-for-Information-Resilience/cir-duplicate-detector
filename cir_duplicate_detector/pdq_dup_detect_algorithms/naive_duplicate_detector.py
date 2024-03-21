import logging

import numpy as np
import pandas as pd

from cir_duplicate_detector.pdq_dup_detect_algorithms.utils import (
    PDQ_HASH_LENGTH,
    calculate_absolute_threshold,
    check_series_empty,
    hamming_distance,
    run_in_parallel,
    validate_similarity_threshold,
)

logger = logging.getLogger(__name__)


def get_single_pdq_fuzzy_duplicates(pdq_hash_series: pd.Series, row_n: int, n: int) -> list[dict[str, str | int]]:
    """Identifies fuzzy duplicates for a single PDQ hash item within the provided series.

    Calculates the Hamming distance between the target PDQ hash and each PDQ hash in the series,
    identifying those within the specified threshold `n`.

    Args:
        pdq_hash_series: Series containing PDQ hash values.
        row_n: Index of the PDQ hash within the series to check against others.
        n: Hamming distance threshold for considering items as duplicates.

    Returns:
        A list of dictionaries, each containing the 'index' of the duplicate item within the series and its
        'dist' (Hamming distance) from the target item.
    """
    pdq_hash = pdq_hash_series.iloc[row_n]
    distances = pdq_hash_series.apply(lambda x: hamming_distance(pdq_hash, x))
    duplicates = distances[distances <= n]
    return [{"index": index, "dist": distance} for index, distance in duplicates.items()]


def get_pdq_fuzzy_duplicates(
    pdq_hash_series: pd.Series, pqd_hash_similarity_threshold: float, indexes_to_check: pd.Series = None
) -> list[list[dict[str, str | int]]]:
    """Retrieves lists of fuzzy duplicate dictionaries for PDQ hashes within a specified similarity threshold.

    This function iterates over `pdq_hash_series`, or optionally only over the indices provided in
    `indexes_to_check`, to find duplicates based on the `pqd_hash_similarity_threshold`.

    Args:
        pdq_hash_series: Series of PDQ hash values to check for duplicates.
        pqd_hash_similarity_threshold: Threshold for considering an item a duplicate, based on similarity.
        indexes_to_check: Optional; Series of indices to specifically check for duplicates. If not provided,
            all indices are checked.

    Returns:
        A nested list where each sublist contains dictionaries of duplicates for a given PDQ hash.
        Each dictionary includes the 'index' of the duplicate and its 'dist' (Hamming distance).
    """
    validate_similarity_threshold(pqd_hash_similarity_threshold)
    if check_series_empty(pdq_hash_series):
        return []

    n = calculate_absolute_threshold(PDQ_HASH_LENGTH, pqd_hash_similarity_threshold)
    if indexes_to_check is not None:
        row_numbers_to_check = list(np.where(pdq_hash_series.index.isin(indexes_to_check))[0])
    else:
        row_numbers_to_check = list(range(len(pdq_hash_series)))

    tasks = [(pdq_hash_series, row_n, n) for row_n in row_numbers_to_check]
    results = run_in_parallel(get_single_pdq_fuzzy_duplicates, tasks)
    return results
