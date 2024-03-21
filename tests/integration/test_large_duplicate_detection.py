import pandas as pd

from cir_duplicate_detector import detect_duplicates


def test_find_large_dataset_duplicates(
    sample_data, expected_output, pqd_hash_similarity_threshold, duplicate_detection_method
):
    # Create a large dataset to test the performance of the duplicate detection
    # and the multi-processing capabilities
    dataset_size = 10000

    extra_dataset_size = dataset_size - len(sample_data)
    # Extend the sample data to create a large dataset
    # Generate random URLs and PDQ hashes
    pdq_hash_steps = 10**60 // extra_dataset_size
    extra_data = {
        "url": [f"https://example.com/{i}" for i in range(1, extra_dataset_size + 1)],
        "pdq_hash": [f"{i*pdq_hash_steps:064x}" for i in range(1, extra_dataset_size + 1)],
    }

    # Add the sample data to the start of the
    larger_df = sample_data
    # Extend the larger_df to create a large dataset
    larger_df = pd.concat([larger_df, pd.DataFrame(extra_data)], ignore_index=True)

    # Make sure the indexes are counted correctly, start at "UW0001" onwards
    larger_df["index"] = [f"UW{i:04}" for i in range(1, len(larger_df) + 1)]
    larger_df = larger_df.set_index("index")

    result = detect_duplicates(
        df=larger_df,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        pdq_duplicate_detection_method=duplicate_detection_method,
    )

    # The extra data is not close the the sample data, thus the expected output stays the same
    pd.testing.assert_frame_equal(result, expected_output)
