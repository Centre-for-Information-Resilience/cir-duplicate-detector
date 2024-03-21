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
    pdq_hash = pdq_hash_series.iloc[row_n]
    distances = pdq_hash_series.apply(lambda x: hamming_distance(pdq_hash, x))
    duplicates = distances[distances <= n]
    return [{"index": index, "dist": distance} for index, distance in duplicates.items()]


def get_pdq_fuzzy_duplicates(
    pdq_hash_series: pd.Series, pqd_hash_similarity_threshold: float, indexes_to_check: pd.Series = None
) -> list[list[dict[str, str | int]]]:
    validate_similarity_threshold(pqd_hash_similarity_threshold)
    if check_series_empty(pdq_hash_series):
        return []
    n = calculate_absolute_threshold(PDQ_HASH_LENGTH, pqd_hash_similarity_threshold)

    if indexes_to_check is not None:
        row_numbers_to_check = np.where(pdq_hash_series.index.isin(indexes_to_check))[0]
    else:
        row_numbers_to_check = range(len(pdq_hash_series))

    tasks = [(pdq_hash_series, row_n, n) for row_n in row_numbers_to_check]
    results = run_in_parallel(get_single_pdq_fuzzy_duplicates, tasks)
    return results


# def pdq_hash_duplicate_detector_worker(args: tuple) -> list[dict[str, str | int]]:
#     """
#     Worker function for the multiprocessing pool. This function is required to be defined outside of the
#     class in order to be picklable.
#     """
#     pdq_hash_series = args[0]
#     row_n = args[1]
#     n = args[2]
#     return get_single_pdq_fuzzy_duplicates(pdq_hash_series=pdq_hash_series, row_n=row_n, n=n)


# def get_single_pdq_fuzzy_duplicates(pdq_hash_series: pd.Series, row_n: int, n: int) -> list[dict[str, str | int]]:
#     """
#     Gets duplicates for a PDQ hash item in the dataframe.

#     It gets the hash from the series at the `row_n` location and check for duplicates in the series.

#     Parameters:
#         pdq_hash_series (pd.Series): Series containing PDQ hash values.
#         row_n (str): The row number of the PDQ hash in the series.
#         n (int): Threshold for considering an item as a duplicate, absolute number.

#     Returns:
#         pd.Series: Series with a column indicating duplicates by index.
#     """
#     pdq_hash = pdq_hash_series.iloc[row_n]

#     distances = pdq_hash_series.apply(lambda x: hamming_distance(pdq_hash, x))
#     duplicates = distances[distances <= n]

#     duplicate_result = [{"index": index, "dist": distance} for index, distance in duplicates.items()]
#     return duplicate_result


# def get_pdq_fuzzy_duplicates(
#     pdq_hash_series: pd.Series,
#     indexes_to_check: pd.Series,
#     pqd_hash_similarity_threshold: float,
# ) -> list[list[dict[str, str | int]]]:
#     """
#     Retrieves duplicates in the series within a given threshold.

#     Parameters:
#         pdq_hash_series (pd.Series): Series containing PDQ hash values
#         indexes_to_check (pd.Series): Series containing the indexes to check
#         pqd_hash_similarity_threshold (float): Threshold (percentage) for considering an item as a duplicate.

#     Returns:
#         list[dict[str, str | int]]: List of dictionaries containing the results from the duplicate detection.
#     """
#     if pqd_hash_similarity_threshold < 0.0 or pqd_hash_similarity_threshold > 1.0:
#         raise ValueError(
#             "pqd_hash_similarity_threshold must be a float between 0.0 and 1.0, " f"got {pqd_hash_similarity_threshold}" #noqa
#         )

#     # Calculate the absolute threshold
#     n = calculate_absolute_threshold(PDQ_HASH_LENGTH, pqd_hash_similarity_threshold)

#     logger.info(f"Using similarity threshold: {pqd_hash_similarity_threshold} (max {n} bits difference).")

#     if len(pdq_hash_series) == 0:
#         warning_text = "The pdq_hash_series is empty, no duplicates will be found."
#         warnings.warn(warning_text, UserWarning, stacklevel=1)
#         logger.warning(warning_text)
#         results = []
#     elif len(pdq_hash_series) == 1:
#         logger.info("The dataframe contains only one row. Running single-threaded.")
#         results = [get_single_pdq_fuzzy_duplicates(pdq_hash_series=pdq_hash_series, row_n=0, n=n)]
#     else:
#         num_cores = cpu_count()

#         start_time = time.time()

#         if indexes_to_check is None:
#             # Check all rows, note that we need to do this on row level, since there can be multiple rows with
#             # the same index
#             row_numbers_to_check = range(len(pdq_hash_series))
#         if indexes_to_check is not None:
#             # Calculate all the row numbers for the indexes to check
#             row_numbers_to_check = np.where(pdq_hash_series.index.isin(indexes_to_check))[0]  # type: ignore

#         # Chunksize of 100 approxamitely results in 100% cpu usage on a 20 core machine
#         target_chunksize = 100
#         multithread_threshold = target_chunksize // 5

#         if len(row_numbers_to_check) < multithread_threshold:
#             # The dataset is small, run it single-threaded without multiprocessing
#             # This is faster than multi-threaded since we do not have to copy the data to each process
#             # This also makes it easier to debug

#             logger.info(f"The dataframe contains less than {multithread_threshold} rows. Running single-threaded.")
#             results = []

#             for row_n in row_numbers_to_check:
#                 results.append(get_single_pdq_fuzzy_duplicates(pdq_hash_series=pdq_hash_series, row_n=row_n, n=n))
#         else:
#             if num_cores > 1:
#                 logger.info(
#                     f"The dataframe contains more than {multithread_threshold} rows. "
#                     f"Running multi-threaded on {num_cores} cores."
#                 )
#             else:
#                 logger.info(
#                     f"The dataframe contains more than {multithread_threshold} rows. "
#                     "Running multi-threaded on 1 core."
#                 )

#             # Multi-threaded
#             with Pool():
#                 chunksize = max(min(target_chunksize, len(row_numbers_to_check) // num_cores), 1)
#                 logger.info(f"Using chunksize: {chunksize}")
#                 results = process_map(
#                     pdq_hash_duplicate_detector_worker,
#                     [(pdq_hash_series, row_n, n) for row_n in row_numbers_to_check],
#                     max_workers=num_cores,
#                     chunksize=chunksize,
#                     smoothing=0,
#                 )

#         end_time = time.time()
#         logger.info(f"Time taken for duplicate check: {round(end_time - start_time, 2)} seconds")

#     return results
