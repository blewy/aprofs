"""Tests for src.aprofs.code """

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from aprofs.code import Aprofs
from aprofs.models import ClassificationLogisticLink
from aprofs.utils import (
    calculate_all_row_sum,
    calculate_row_sum,
)

data_dir = Path(__file__).parent.parent / "docs"  # Navigate up to the project root
data_path = data_dir / "insurance.csv"


@pytest.fixture
def link_model():
    return ClassificationLogisticLink()


@pytest.mark.parametrize(
    "data, features, expected_value, expected_output",
    [
        (
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "s": [3, 4]}),
            ["a", "b"],
            1,
            pd.Series([1 / (1 + np.exp(-5)), 1 / (1 + np.exp(-7))]),
        ),
    ],
)
def test__calculate_row_sum(data, features, expected_value, link_model, expected_output):
    assert calculate_row_sum(data, expected_value, columns=features, link_model=link_model).equals(expected_output)


@pytest.mark.parametrize(
    "data, expected_value, expected_output",
    [
        (
            pd.DataFrame({"a": [1, 2], "b": [3, 4], "s": [2, 5]}),
            1,
            pd.Series([1 / (1 + np.exp(-7)), 1 / (1 + np.exp(-12))]),
        ),
    ],
)
def test__calculate_all_row_sum(data, expected_value, link_model, expected_output):
    assert calculate_all_row_sum(data, expected_value, link_model=link_model).equals(expected_output)


@pytest.fixture
def features():
    return [
        "age",
        "bmi",
        "children",
        "smoker",
        "region",
        "charges",
    ]


def tutorial_data(features):
    data = pd.read_csv(data_path)
    # foor loop over a pandas dataframe columns and chnate the typos off all string columns to category
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype("category")
    # set target
    data["is_female"] = (data["sex"] == "female").astype(int)  # is female becomes the target variable
    data = data.drop(columns=["sex"])

    seed = 42
    data_x, y_target = data[features], data["is_female"]
    return train_test_split(data_x, y_target, stratify=y_target, random_state=seed)


def tutorial_model(features):
    x_train, x_valid, y_train, y_valid = tutorial_data(features)

    monotone_constraints = [1 if col == "charges" else 0 for col in features]

    callbacks = [lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)]
    seed = 42
    model = LGBMClassifier(
        verbose=-1, n_estimators=100, monotone_constraints=monotone_constraints, random_state=seed
    ).fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        callbacks=callbacks,
    )
    return model


def test__object_creation(features):
    _, x_data, _, y_data = tutorial_data(features)
    aprofs_objct = Aprofs(x_data, y_data, link_model=None)
    assert not aprofs_objct.current_data.empty
    assert not aprofs_objct.target_column.empty


def test__shap_calculation(features):
    # get data
    _, x_data, _, y_data = tutorial_data(features)
    # get model
    model_ = tutorial_model(features)
    # crete object
    aprofs_objct = Aprofs(x_data, y_data, link_model=None)
    aprofs_objct.calculate_shaps(model_)
    assert not aprofs_objct.shap_values.empty
    assert aprofs_objct.shap_mean is not None
    assert not aprofs_objct.shap_mean == 0


def test__get_performance(features):
    # get data
    _, x_data, _, y_data = tutorial_data(features)
    # get model
    model_ = tutorial_model(features)
    # crete object
    aprofs_objct = Aprofs(x_data, y_data, link_model=None)
    aprofs_objct.calculate_shaps(model_)
    perf = aprofs_objct.get_feature_performance(features)
    assert perf is not None
    assert not perf == 0


def test__get_brute_features(features):
    # get data
    _, x_data, _, y_data = tutorial_data(features)
    # get model
    model_ = tutorial_model(features)
    # crete object
    aprofs_objct = Aprofs(x_data, y_data, link_model=None)
    aprofs_objct.calculate_shaps(model_)
    test_features = aprofs_objct.brute_force_selection(features)
    assert test_features, "test_features is not empty"


def test__get_gready_features(features):
    # get data
    _, x_data, _, y_data = tutorial_data(features)
    # get model
    model_ = tutorial_model(features)
    # crete object
    aprofs_objct = Aprofs(x_data, y_data, link_model=None)
    aprofs_objct.calculate_shaps(model_)
    test_features = aprofs_objct.gready_forward_selection(features)
    assert test_features, "test_features is not empty"
