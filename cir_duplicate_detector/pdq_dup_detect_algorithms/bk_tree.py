import collections
import logging
import time

import pandas as pd
import pybktree

from cir_duplicate_detector.pdq_dup_detect_algorithms.utils import (
    PDQ_HASH_LENGTH,
    calculate_absolute_threshold,
    check_series_empty,
    hamming_distance,
    run_in_parallel,
    validate_similarity_threshold,
)

logger = logging.getLogger(__name__)

Item = collections.namedtuple("Item", "bits id")


class PDQHashTree:
    """A class for efficient similarity search of PDQ hashes using a BK-tree structure."""

    def __init__(self, pdq_hash_series: pd.Series):
        """Initializes the PDQHashTree with a series of PDQ hashes.

        Args:
            pdq_hash_series: A pandas Series containing PDQ hash values.
        """
        self.tree = self._build_bktree(pdq_hash_series)

    def _item_distance_function(self, item1: Item, item2: Item) -> int:
        """Computes the Hamming distance between two items.

        Args:
            item1: The first item.
            item2: The second item.

        Returns:
            The Hamming distance between the two items.
        """
        return hamming_distance(item1.bits, item2.bits)

    def _build_bktree(self, pdq_hash_series: pd.Series) -> pybktree.BKTree:
        """Builds a BK-tree from the provided series of PDQ hashes.

        Args:
            pdq_hash_series: A pandas Series containing PDQ hash values.

        Returns:
            A BKTree object populated with items from the pdq_hash_series.
        """
        start_time = time.time()
        items = [Item(value, index) for index, value in pdq_hash_series.items()]
        tree = pybktree.BKTree(self._item_distance_function, items)
        logger.info(
            f"Time taken to build BK-tree containing {len(items)} nodes: {round(time.time() - start_time, 2)} seconds"
        )
        return tree

    def _get_pdq_duplicates(self, index: str, pdq_hash: str, n: int) -> list[dict[str, str | int]]:
        """Finds PDQ hash duplicates for a given hash within a threshold.

        Args:
            index: The index of the PDQ hash in the series.
            pdq_hash: The PDQ hash to check for duplicates.
            n: The maximum allowed Hamming distance for duplicates.

        Returns:
            A list of dictionaries, each representing a duplicate. The dictionaries contain the 'index' of the
            duplicate and its 'dist' (Hamming distance).
        """
        pdq_item = Item(pdq_hash, index)
        duplicates = self.tree.find(item=pdq_item, n=n)
        duplicates = [item for item in duplicates if item[1].id != pdq_item.id]
        return [{"index": item[1].id, "dist": item[0]} for item in duplicates]

    def get_duplicates(
        self, pdq_hash_series: pd.Series, pqd_hash_similarity_threshold: float
    ) -> list[list[dict[str, str | int]]]:
        """Finds all PDQ hash duplicates in the provided series within a similarity threshold.

        Args:
            pdq_hash_series: A pandas Series containing PDQ hash values.
            pqd_hash_similarity_threshold: The similarity threshold for considering an item a duplicate.

        Returns:
            A list of lists, where each sublist contains dictionaries of duplicates for a given PDQ hash.
        """
        validate_similarity_threshold(pqd_hash_similarity_threshold)
        if check_series_empty(pdq_hash_series):
            return []
        n = calculate_absolute_threshold(PDQ_HASH_LENGTH, pqd_hash_similarity_threshold)
        tasks = [(index, pdq_hash, n) for index, pdq_hash in pdq_hash_series.items()]
        results = run_in_parallel(self._get_pdq_duplicates, tasks)
        return results
