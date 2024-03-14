import pandas as pd
import pytest

from cir_duplicate_detector.url import extract_base_url, find_url_duplicates


@pytest.fixture
def url_series(sample_data):
    return sample_data["url"]


@pytest.fixture
def expected_output(expected_output):
    return expected_output["url_duplicates"].dropna()


def test_find_url_duplicates(url_series, expected_output):
    url_series_original = url_series.copy()
    # Normal flow
    result = find_url_duplicates(url_series=url_series)

    pd.testing.assert_series_equal(result, expected_output)

    # Make sure the original series is not modified
    pd.testing.assert_series_equal(url_series, url_series_original)


def test_extract_base_url():
    urls = [
        "https://example.com/path;param1=val1?arg=value&arg2=value2#15213",
        "http://hello.example.com/otherpath#page=123",
        "https://example.com/path?differentarg=value#yes=no",
        "https://test123.anotherexample.com/path2?arg=value#42",
        "https://EXAMPLE.com/Path?arg=value",
    ]

    expected_output = [
        "example.com/path;param1=val1?arg=value&arg2=value2",
        "hello.example.com/otherpath",
        "example.com/path?differentarg=value",
        "test123.anotherexample.com/path2?arg=value",
        "example.com/path?arg=value",
    ]

    for url, expected in zip(urls, expected_output, strict=True):
        base_url = extract_base_url(url)
        assert base_url == expected


def test_missing_index(url_series):
    # Remove the index
    url_series = url_series.reset_index(drop=True)

    with pytest.raises(ValueError):
        find_url_duplicates(url_series=url_series)


def test_empty_url_series(url_series):
    # Create an empty url series with the same index name
    s = pd.Series([], name=url_series.name)
    s.index.name = url_series.index.name

    # Check that a warning is logged
    # Check that the result is an empty series
    with pytest.warns(UserWarning):
        result = find_url_duplicates(url_series=s)

    expected_result = pd.Series([], name="url_duplicates", dtype=object)
    expected_result.index.name = url_series.index.name

    pd.testing.assert_series_equal(result, expected_result)


def test_missing_url(url_series, expected_output):
    # Remove the url from the url_series
    url_series = url_series.drop(index="UW0001")

    # Remove all mentions of UW0001 from the expected output
    expected_output = expected_output.drop(index="UW0001")
    for _, duplicates in expected_output.items():
        if not isinstance(duplicates, type(pd.NA)) and "UW0001" in duplicates:
            duplicates.remove("UW0001")

    # Everything should work fine
    result = find_url_duplicates(url_series=url_series)

    pd.testing.assert_series_equal(result, expected_output)


def test_na_url(url_series, expected_output):
    # Set the url of UW0001 to NA
    url_series["UW0001"] = pd.NA

    # Remove all mentions of UW0001 from the expected output
    expected_output = expected_output.drop(index="UW0001")
    for _, duplicates in expected_output.items():
        if not isinstance(duplicates, type(pd.NA)) and "UW0001" in duplicates:
            duplicates.remove("UW0001")

    # Everything should work fine
    result = find_url_duplicates(url_series=url_series)

    pd.testing.assert_series_equal(result, expected_output)


def test_indexes_to_check(url_series, expected_output):
    # Create a indexes to check list
    indexes_to_check = ["UW0004", "UW0005"]

    # Remove all duplicates from the first three indexes since they do not share any duplicates with
    # the indexes to check
    expected_output["UW0001"] = pd.NA
    expected_output["UW0002"] = pd.NA
    expected_output["UW0003"] = pd.NA

    expected_output = expected_output.dropna()

    result = find_url_duplicates(url_series=url_series, indexes_to_check=indexes_to_check)

    pd.testing.assert_series_equal(result, expected_output)


def test_result_not_containing_nan(url_series, expected_output):
    result = find_url_duplicates(url_series=url_series)

    assert not result.isna().to_numpy().any()


def test_no_duplicates(url_series, expected_output):
    # Remove all duplicates from the url_series
    url_series = url_series[["UW0001", "UW0004"]]

    expected_output = expected_output.iloc[0:0]

    # Everything should work fine
    result = find_url_duplicates(url_series=url_series)

    pd.testing.assert_series_equal(result, expected_output)
