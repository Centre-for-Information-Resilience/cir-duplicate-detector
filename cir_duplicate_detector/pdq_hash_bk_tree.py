import collections
import logging
import time
import warnings
from multiprocessing import Pool, cpu_count

import pandas as pd
import pybktree
import rapidfuzz
from tqdm.contrib.concurrent import process_map

logger = logging.getLogger(__name__)

PDQ_HASH_LENGTH = 256

Item = collections.namedtuple("Item", "bits id")


def pdq_hash_tree_duplicate_detector_worker(args: tuple) -> list[dict[str, str | int]]:
    """
    Worker function for the multiprocessing pool. This function is required to be defined outside of the
    class in order to be picklable.
    """
    pdq_hash_tree = args[0]
    return pdq_hash_tree._get_pdq_duplicates(*args[1:])


class PDQHashTree:
    """
    Represents a BK-tree of PDQ hash items for efficient similarity search.

    Attributes:
        tree (pybktree.BKTree): BK-tree structure for searching similar PDQ hash items.

    Methods:
        __init__(self, pdq_hash_series): Constructs the BK-tree from a series.
        get_all_duplicates(self, pdq_hashes_to_check, n): Retrieves all duplicates in the series within a
        given threshold.
    """

    def __init__(self, pdq_hash_series: pd.Series) -> None:
        """
        Constructs the BK-tree from a series.

        Parameters:
            pdq_hash_series (pd.Series): Series containing PDQ hash values.
        """
        self.tree = self._build_bktree(pdq_hash_series)

    @staticmethod
    def _item_distance_function(item1: Item, item2: Item) -> int:
        """
        [Internal] Computes the Hamming distance between two PDQ hash items.

        Parameters:
            item1 (Item): First PDQ hash item.
            item2 (Item): Second PDQ hash item.

        Returns:
            int: Hamming distance between item1 and item2.
        """

        # It seems unintuitive that the Hamming distance is computed on the binary strings
        # However, after trying to use the hamming distance on interegers, numpy arrays and bytes
        # using python hammingsdistance, numpy hammingsdistance and pybktree hammingdistance
        # it seems that rapidfuzz is orders of magnitude faster than the other methods
        return rapidfuzz.distance.Hamming.distance(item1.bits, item2.bits)

    def _build_bktree(self, pdq_hash_series: pd.Series) -> pybktree.BKTree:
        """
        [Internal] Builds a BK-tree from a series.

        Parameters:
            pdq_hash_series (pd.Series): Series containing PDQ hash values.

        Returns:
            pybktree.BKTree: BK-tree for PDQ hash items.
        """
        start_time = time.time()

        # Build the BK-tree
        items = [Item(value, index) for index, value in pdq_hash_series.items()]
        tree = pybktree.BKTree(self._item_distance_function, items)

        end_time = time.time()
        logger.info(
            f"Time taken to build BK-tree containing {len(items)} nodes: {round(end_time - start_time, 2)} seconds"
        )
        return tree

    def _get_pdq_item(self, pdq_hash_series: pd.Series, index: str) -> Item:
        """
        [Internal] Retrieves a PDQ hash item from a series.

        Parameters:
            pdq_hash_series (pd.Series): Series containing PDQ hash values.
            index (str): Index of the item in the series.

        Returns:
            Item: PDQ hash item.
        """
        return Item(pdq_hash_series[index], index)

    def _get_pdq_duplicates(self, index: str, pdq_hash: str, n: int) -> list[dict[str, str | int]]:
        """
        [Internal] Gets duplicates for a PDQ hash item in the dataframe.

        Parameters:
            index (str): The index of the PDQ hash in the series.
            pdq_hash (str): The PDQ hash to check for duplicates.
            n (int): Threshold for considering an item as a duplicate, absolute number.

        Returns:
            pd.Series: Series with a column indicating duplicates by index.
        """

        # Get the PDQ hash item used in the BK-tree
        pdq_item = Item(pdq_hash, index)

        # Search the BK-tree for duplicates, sort the results by distance
        duplicates = self.tree.find(item=pdq_item, n=n)
        duplicates = [item for item in duplicates if item[1].id != pdq_item.id]

        duplicate_result = [{"index": item[1].id, "dist": item[0]} for item in duplicates]
        return duplicate_result

    def get_duplicates(
        self, pdq_hash_series: pd.Series, pqd_hash_similarity_threshold: float
    ) -> list[list[dict[str, str | int]]]:
        """
        Retrieves duplicates in the series within a given threshold.

        Parameters:
            pdq_hash_series (pd.Series): Series containing PDQ hash values, this can be a subset or different
                series than the one used to construct the BK-tree.
            pqd_hash_similarity_threshold (float): Threshold (percentage) for considering an item as a duplicate.

        Returns:
            list[dict[str, str | int]]: List of dictionaries containing the results from the duplicate detection.
        """

        if pqd_hash_similarity_threshold < 0.0 or pqd_hash_similarity_threshold > 1.0:
            raise ValueError(
                "pqd_hash_similarity_threshold must be a float between 0.0 and 1.0, "
                f"got {pqd_hash_similarity_threshold}"
            )

        # Calculate the absolute threshold
        n = int(round(PDQ_HASH_LENGTH * (1 - pqd_hash_similarity_threshold)))

        logger.info(f"Using similarity threshold: {pqd_hash_similarity_threshold} (max {n} bits difference).")

        if len(pdq_hash_series) == 0:
            warning_text = "The pdq_hash_series is empty, no duplicates will be found."
            warnings.warn(warning_text, UserWarning, stacklevel=1)
            logger.warning(warning_text)
            results = []
        elif len(pdq_hash_series) == 1:
            logger.info("The dataframe contains only one row. Running single-threaded.")
            results = [
                self._get_pdq_duplicates(
                    index=pdq_hash_series.index[0],
                    pdq_hash=pdq_hash_series.iloc[0],
                    n=n,
                )
            ]
        else:
            num_cores = cpu_count()

            start_time = time.time()

            # Chunksize of 100 approxamitely results in 100% cpu usage on a 20 core machine
            target_chunksize = 100
            multithread_threshold = target_chunksize // 5

            if len(pdq_hash_series) < multithread_threshold:
                # The dataset is small, run it single-threaded without multiprocessing
                # This is faster than multi-threaded since we do not have to copy the data to each process
                # This also makes it easier to debug

                logger.info(f"The dataframe contains less than {multithread_threshold} rows. Running single-threaded.")
                results = []
                for index, pdq_hash in pdq_hash_series.items():
                    results.append(
                        self._get_pdq_duplicates(
                            index=index,
                            pdq_hash=pdq_hash,
                            n=n,
                        )
                    )
            else:
                if num_cores > 1:
                    logger.info(
                        f"The dataframe contains more than {multithread_threshold} rows. "
                        f"Running multi-threaded on {num_cores} cores."
                    )
                else:
                    logger.info(
                        f"The dataframe contains more than {multithread_threshold} rows. "
                        "Running multi-threaded on 1 core."
                    )

                # Multi-threaded
                with Pool():
                    chunksize = max(min(target_chunksize, len(pdq_hash_series) // num_cores), 1)
                    logger.info(f"Using chunksize: {chunksize}")
                    results = process_map(
                        pdq_hash_tree_duplicate_detector_worker,
                        [(self, pdq_hash, index, n) for pdq_hash, index in pdq_hash_series.items()],
                        max_workers=num_cores,
                        chunksize=chunksize,
                        smoothing=0,
                    )

            end_time = time.time()
            logger.info(f"Time taken for duplicate check: {round(end_time - start_time, 2)} seconds")

        return results
