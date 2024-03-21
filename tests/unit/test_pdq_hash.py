import pandas as pd
import pytest

from cir_duplicate_detector.pdq_hash import find_pdq_hash_duplicates


@pytest.fixture
def pdq_hash_series(sample_data):
    return sample_data["pdq_hash"]


@pytest.fixture
def expected_output(expected_output):
    return expected_output[["pdq_hash_duplicates", "pdq_hash_similarities"]].dropna(how="all")


# Runn all tests in this file with both duplicate_detection_methods
@pytest.fixture(params=["naive", "bk-tree"])  # TODO: add mih
def duplicate_detection_method(request):
    return request.param


def test_find_pdq_hash_duplicates(
    pdq_hash_series,
    expected_output,
    pqd_hash_similarity_threshold,
    duplicate_detection_method,
):
    # Test the default case
    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        duplicate_detection_method=duplicate_detection_method,
    )

    pd.testing.assert_frame_equal(result, expected_output)


def test_with_empty_pdq_hash_series(
    pdq_hash_series,
    expected_output,
    pqd_hash_similarity_threshold,
    duplicate_detection_method,
):
    # Test with empty pdq_hash_series
    pdq_hash_series = pdq_hash_series.iloc[0:0]

    # Check that a warning is raised
    with pytest.warns(UserWarning):
        result = find_pdq_hash_duplicates(
            pdq_hash_series=pdq_hash_series,
            pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
            duplicate_detection_method=duplicate_detection_method,
        )

    expected_output = expected_output.iloc[0:0]

    pd.testing.assert_frame_equal(result, expected_output)


def test_pdq_hash_series_containing_nan(
    pdq_hash_series,
    expected_output,
    pqd_hash_similarity_threshold,
    duplicate_detection_method,
):
    # Replace a pdq hash with an Nan value, everything should work fine
    pdq_hash_series.iloc[0] = pd.NA
    pdq_hash_series.iloc[1] = None
    pdq_hash_series.iloc[2] = float("nan")

    expected_output.iloc[0] = pd.NA
    expected_output.iloc[1] = pd.NA
    expected_output.iloc[2] = pd.NA
    expected_output = expected_output.dropna()

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        duplicate_detection_method=duplicate_detection_method,
    )

    pd.testing.assert_frame_equal(result, expected_output)


def test_missing_index(pdq_hash_series):
    # Remove the index
    pdq_hash_series = pdq_hash_series.reset_index(drop=True)

    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series)


def test_missing_pqd_hash_similarity_threshold(pdq_hash_series):
    # The pdq hash similarity should default to a number if not provided
    result = find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series)

    # Check that the result contains some duplicates
    assert len(result) > 0


def test_invalid_pqd_hash_similarity_threshold(pdq_hash_series):
    # Test with invalid pdq_hash_similarity_threshold
    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold=-0.1)

    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold="abc")

    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold=None)


def test_too_large_pqd_hash_similarity_threshold(pdq_hash_series):
    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold=1.1)


def test_too_small_pqd_hash_similarity_threshold(pdq_hash_series):
    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series, pqd_hash_similarity_threshold=-0.1)


def test_invalid_pdq_hash_series(pdq_hash_series):
    # Replace a pdq hash with an invalid value
    pdq_hash_series.iloc[1] = 123

    with pytest.raises(ValueError):
        find_pdq_hash_duplicates(pdq_hash_series=pdq_hash_series)


def test_different_hash_lengths(
    pdq_hash_series,
    expected_output,
    pqd_hash_similarity_threshold,
    duplicate_detection_method,
):
    # Increase the size of the 6'th hash, this will affect the relative hamming distance and thus the
    # result. The similarity will be smaller
    pdq_hash_series.iloc[5][0] = pdq_hash_series.iloc[5][0] + "00000000000"

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        duplicate_detection_method=duplicate_detection_method,
    )

    # Only check the duplicates column
    result = result["pdq_hash_duplicates"]
    expected_output = expected_output["pdq_hash_duplicates"]

    pd.testing.assert_series_equal(result, expected_output)


def test_no_duplicates(pdq_hash_series, expected_output, duplicate_detection_method):
    # Select the rows that are non identical
    pdq_hash_series = pdq_hash_series.iloc[2:]

    # Set the pdq hash similarity threshold to 1.0
    pqd_hash_similarity_threshold = 1.0

    # The expected output is an empty series
    expected_output = expected_output.iloc[0:0]

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        duplicate_detection_method=duplicate_detection_method,
    )

    pd.testing.assert_frame_equal(result, expected_output)


def test_with_indexes_to_check(pdq_hash_series, pqd_hash_similarity_threshold, duplicate_detection_method):
    # Test with indexes_to_check
    indexes_to_check = ["UW0001", "UW0005"]

    expected_duplicates = {
        "UW0001": ["UW0002", "UW0003"],
        "UW0002": ["UW0001"],
        "UW0003": ["UW0001"],
        "UW0004": ["UW0005"],
        "UW0005": ["UW0004"],
    }

    expected_similarities = {
        "UW0001": [1.0, 1.0],
        "UW0002": [1.0],
        "UW0003": [1.0],
        "UW0004": [0.98828125],
        "UW0005": [0.98828125],
    }

    expected_output = pd.DataFrame(
        {
            "pdq_hash_duplicates": pd.Series(expected_duplicates, dtype=object),
            "pdq_hash_similarities": pd.Series(expected_similarities, dtype=object),
        }
    )
    expected_output.index.name = "index"

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        indexes_to_check=indexes_to_check,
        duplicate_detection_method=duplicate_detection_method,
    )

    print(f"result: {result}")
    print(f"expected_output: {expected_output}")

    pd.testing.assert_frame_equal(result, expected_output)


def test_single_index_to_check(pdq_hash_series, pqd_hash_similarity_threshold, duplicate_detection_method):
    # Test with a single index_to_check
    indexes_to_check = ["UW0001"]

    expected_duplicates = {
        "UW0001": ["UW0002", "UW0003"],
        "UW0002": ["UW0001"],
        "UW0003": ["UW0001"],
    }

    expected_similarities = {
        "UW0001": [1.0, 1.0],
        "UW0002": [1.0],
        "UW0003": [1.0],
    }

    expected_output = pd.DataFrame(
        {
            "pdq_hash_duplicates": pd.Series(expected_duplicates, dtype=object),
            "pdq_hash_similarities": pd.Series(expected_similarities, dtype=object),
        }
    )
    expected_output.index.name = "index"

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        indexes_to_check=indexes_to_check,
        duplicate_detection_method=duplicate_detection_method,
    )

    pd.testing.assert_frame_equal(result, expected_output)


def test_with_empty_indexes_to_check(
    pdq_hash_series,
    expected_output,
    pqd_hash_similarity_threshold,
    duplicate_detection_method,
):
    # Test with empty indexes_to_check
    indexes_to_check = pdq_hash_series.index[0:0]

    with pytest.warns(UserWarning):
        result = find_pdq_hash_duplicates(
            pdq_hash_series=pdq_hash_series,
            pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
            indexes_to_check=indexes_to_check,
            duplicate_detection_method=duplicate_detection_method,
        )

    expected_output = expected_output.iloc[0:0]

    pd.testing.assert_frame_equal(result, expected_output)


def test_index_to_check_with_empty_hash(pdq_hash_series, pqd_hash_similarity_threshold, duplicate_detection_method):
    # Test with a index_to_check that has an empty hash
    indexes_to_check = ["UW0001", "UW0002"]

    pdq_hash_series["UW0002"] = pd.NA

    expected_duplicates = {
        "UW0001": ["UW0003"],
        "UW0003": ["UW0001"],
    }
    expected_similarities = {
        "UW0001": [1.0],
        "UW0003": [1.0],
    }

    expected_output = pd.DataFrame(
        {
            "pdq_hash_duplicates": pd.Series(expected_duplicates, dtype=object),
            "pdq_hash_similarities": pd.Series(expected_similarities, dtype=object),
        }
    )
    expected_output.index.name = "index"

    result = find_pdq_hash_duplicates(
        pdq_hash_series=pdq_hash_series,
        pqd_hash_similarity_threshold=pqd_hash_similarity_threshold,
        indexes_to_check=indexes_to_check,
        duplicate_detection_method=duplicate_detection_method,
    )

    pd.testing.assert_frame_equal(result, expected_output)
