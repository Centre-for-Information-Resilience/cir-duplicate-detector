import pandas as pd
import pytest

from cir_duplicate_detector.utils import drop_literal_series_duplicates, hex_to_binary


@pytest.fixture
def sample_series():
    df = pd.DataFrame(
        {
            "index": [
                "1",
                "2",
                "2",
                "3",
                "4",
                "5",
            ],
            "data": [
                "a",
                "b",  # literal duplicate to iloc 3
                "b",  # literal duplicate to iloc 2
                "b",  # data duplicate to iloc 2
                "c",
                "a",  # data duplicate to iloc 0
            ],
        }
    )

    return df.set_index("index")["data"]


@pytest.fixture
def expected_series_output():
    df = pd.DataFrame(
        {
            "index": [
                "1",
                "2",
                "3",
                "4",
                "5",
            ],
            "data": [
                "a",
                "b",
                "b",
                "c",
                "a",
            ],
        }
    )

    return df.set_index("index")["data"]


def test_drop_literal_series_duplicates(sample_series, expected_series_output):
    # Test normal flow
    result = drop_literal_series_duplicates(sample_series)

    pd.testing.assert_series_equal(result, expected_series_output)


def test_drop_literal_series_duplicates_empty_series():
    # Test empty series
    series = pd.Series(dtype="object")
    result = drop_literal_series_duplicates(series)

    pd.testing.assert_series_equal(result, series)


def test_drop_literal_series_duplicates_index_not_set(sample_series, expected_series_output):
    # Test index not set
    sample_series.index.name = None
    expected_series_output.index.name = None

    result = drop_literal_series_duplicates(sample_series)

    pd.testing.assert_series_equal(result, expected_series_output)


def test_drop_literal_series_duplicates_contains_nan(sample_series, expected_series_output):
    # Test contains nan
    sample_series["6"] = pd.NA
    expected_series_output["6"] = pd.NA

    result = drop_literal_series_duplicates(sample_series)

    pd.testing.assert_series_equal(result, expected_series_output)


def test_drop_literal_series_duplicates_better_than_pd_dropna_function(sample_series, expected_series_output):
    # Test that the function is better than pd.dropna()
    # If pd.dropna() works as well we should use that
    result = sample_series.dropna()

    with pytest.raises(AssertionError):
        pd.testing.assert_series_equal(result, expected_series_output)


def test_hex_to_binary():
    # Test normal flow
    hex_string = "FF"
    expected_binary_string = "11111111"
    result = hex_to_binary(hex_string)
    assert result == expected_binary_string

    # Test with specific length
    hex_string = "A"
    expected_binary_string = "1010"
    result = hex_to_binary(hex_string, length=4)
    assert result == expected_binary_string

    # Test with longer hex string
    hex_string = "ABCDEF"
    expected_binary_string = "101010111100110111101111"
    result = hex_to_binary(hex_string)
    assert result == expected_binary_string

    # Test with trailing zeros (pqd hash black image)
    hex_string = "1134000011342c4b00002c4b1134000000002c4b2c4b00002c4b8200554b"
    expected_binary_string = "0000000000000000000100010011010000000000000000000001000100110100001011000100101100000000000000000010110001001011000100010011010000000000000000000000000000000000001011000100101100101100010010110000000000000000001011000100101110000010000000000101010101001011"  # noqa: E501
    result = hex_to_binary(hex_string, length=256)

    # Test with shorter hex string
    hex_string = "1"
    expected_binary_string = "1"
    result = hex_to_binary(hex_string)
    assert result == expected_binary_string

    # Test with zero hex string
    hex_string = "0"
    expected_binary_string = "0"
    result = hex_to_binary(hex_string)
    assert result == expected_binary_string

    # Test with empty hex string
    hex_string = ""
    expected_binary_string = ""
    result = hex_to_binary(hex_string)
    assert result == expected_binary_string
