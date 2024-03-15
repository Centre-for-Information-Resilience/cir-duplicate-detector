import pandas as pd
import pytest

from cir_duplicate_detector import detect_duplicates


def test_find_duplicates(sample_data, expected_output, pqd_hash_similarity_threshold):
    data = sample_data

    result = detect_duplicates(df=data, pqd_hash_similarity_threshold=pqd_hash_similarity_threshold)

    pd.testing.assert_frame_equal(result, expected_output)


def test_find_duplicates_url_only(sample_data, expected_output):
    data = sample_data["url"].to_frame()
    expected_output = expected_output["url_duplicates"].dropna().to_frame()
    # Insert the index column at the beginning
    expected_output.insert(0, "index", expected_output.index)

    result = detect_duplicates(data)

    pd.testing.assert_frame_equal(result, expected_output)


def test_find_duplicates_pdq_hash_only(sample_data, expected_output, pqd_hash_similarity_threshold):
    data = sample_data["pdq_hash"].to_frame()
    expected_output = expected_output[["pdq_hash_duplicates", "pdq_hash_similarities"]].dropna()
    # Insert the index column at the beginning
    expected_output.insert(0, "index", expected_output.index)

    result = detect_duplicates(df=data, pqd_hash_similarity_threshold=pqd_hash_similarity_threshold)

    pd.testing.assert_frame_equal(result, expected_output)


def test_find_duplicates_with_index_column(sample_data, expected_output, pqd_hash_similarity_threshold):
    index = sample_data.index

    # Add the index as a column
    sample_data["index"] = index

    # Reset the index
    sample_data = sample_data.reset_index(drop=True)

    # Call the function
    result = detect_duplicates(df=sample_data, pqd_hash_similarity_threshold=pqd_hash_similarity_threshold)

    # We expect the same result as before
    pd.testing.assert_frame_equal(result, expected_output)


def test_find_duplicates_with_missing_columns(sample_data):
    # Remove the 'url' and 'pdq_hash' columns
    sample_data = sample_data.drop(columns=["url", "pdq_hash"])

    # Call the function
    with pytest.raises(ValueError):
        detect_duplicates(sample_data)


def test_find_duplicates_with_missing_index(sample_data):
    # Remove the index
    sample_data = sample_data.reset_index(drop=True)

    # Call the function
    with pytest.raises(ValueError):
        detect_duplicates(sample_data)


def test_find_duplicates_empty_dataframe(sample_data, expected_output):
    df = sample_data.iloc[0:0]

    expected_output = expected_output.iloc[0:0]

    with pytest.warns(UserWarning):
        result = detect_duplicates(df)

    pd.testing.assert_frame_equal(result, expected_output)


def test_non_unique_indexes(sample_data):
    # Clone the first row
    sample_data = pd.concat([sample_data, sample_data.iloc[0:1]], ignore_index=False)

    # Throw a error if the index is not unique
    with pytest.raises(ValueError):
        detect_duplicates(sample_data)


def test_url_or_pdq_missing(sample_data, expected_output):
    # Select only the first two rows
    sample_data = sample_data.loc[["UW0001", "UW0002"]]

    sample_data.loc["UW0001", "url"] = pd.NA
    sample_data.loc["UW0002", "pdq_hash"] = pd.NA

    # We expect an empty dataframe as output
    expected_output = expected_output.iloc[0:0]

    result = detect_duplicates(sample_data)

    pd.testing.assert_frame_equal(result, expected_output)


def test_index_to_check_not_in_dataframe(sample_data, expected_output):
    sample_data = sample_data.loc[["UW0001", "UW0002", "UW0003"]]

    # Remove all data from UW0002 but keep the row
    sample_data.loc["UW0002", "url"] = pd.NA
    sample_data.loc["UW0002", "pdq_hash"] = pd.NA

    indexes_to_check = ["UW0002"]

    # We expect an empty dataframe as output
    expected_output = expected_output.iloc[0:0]

    result = detect_duplicates(sample_data, indexes_to_check=indexes_to_check)

    pd.testing.assert_frame_equal(result, expected_output)
