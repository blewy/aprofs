"""Tests for src.aprofs.code """

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aprofs.utils import (
    calculate_all_row_sum,
    calculate_row_sum,
    link_function,
)

data_dir = Path(__file__).parent.parent / "docs"  # Navigate up to the project root
data_path = data_dir / "insurance.csv"


@pytest.mark.parametrize(
    "link, input_value, expected_output",
    [
        ("logistic", 0, 0.5),
        ("logistic", 1, 1 / (1 + np.exp(-1))),
        ("logarithmic", 0, 1),
        ("logarithmic", 1, np.exp(1)),
        ("identity", -2, -2),
        ("identity", 0, 0),
    ],
)
def test__link_function(link, input_value, expected_output):
    func = link_function(link)
    assert np.isclose(func(input_value), expected_output, atol=1e-6)


def test__link_function_invalid_link():
    with pytest.raises(ValueError):
        link_function("invalid_link")


@pytest.mark.parametrize(
    "data, features, expected_value, link, expected_output",
    [
        (pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4]}), ["a", "b"], 0, "identity", pd.Series([4, 6])),
        (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), ["a", "b"], 1, "identity", pd.Series([5, 7])),
        (
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "s": [3, 4]}),
            ["a", "b"],
            1,
            "logistic",
            pd.Series([1 / (1 + np.exp(-5)), 1 / (1 + np.exp(-7))]),
        ),
    ],
)
def test__calculate_row_sum(data, features, expected_value, link, expected_output):
    assert calculate_row_sum(data, expected_value, columns=features, link_function=link_function(link)).equals(
        expected_output
    )


@pytest.mark.parametrize(
    "data, expected_value, link, expected_output",
    [
        (pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [3, 4]}), 0, "identity", pd.Series([7, 10])),
        (pd.DataFrame({"a": [1, 2], "b": [3, 4]}), 1, "identity", pd.Series([5, 7])),
        (
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "s": [2, 5]}),
            1,
            "logistic",
            pd.Series([1 / (1 + np.exp(-7)), 1 / (1 + np.exp(-12))]),
        ),
    ],
)
def test__calculate_all_row_sum(data, expected_value, link, expected_output):
    assert calculate_all_row_sum(data, expected_value, link_function=link_function(link)).equals(expected_output)
