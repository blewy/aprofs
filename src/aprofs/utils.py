"""Utility functions for the package.
this module contains utility functions that are used in the package.
the core functions are used to calculate the SHAP values and expected average SHAP value for a given dataset and model.
"""

from itertools import combinations
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shap import TreeExplainer
from sklearn.metrics import roc_auc_score


def link_function(link: str) -> Callable:
    """
    Returns a link function based on the specified link type.

    Args:
        link (str): The type of link function to use. Must be one of the following:
            - 'logistic': Returns the logistic function 1 / (1 + exp(-shap_sum)).
            - 'logarithmic': Returns the exponential function exp(shap_sum).
            - 'identity': Returns the input shap_sum as is.

    Returns:
        function: The link function corresponding to the specified link type.

    Raises:
        ValueError: If an invalid link type is specified.

    Examples:
        >>> link_func = link_function('logistic')
        >>> link_func(0)
        0.5

        >>> link_func = link_function('logarithmic')
        >>> link_func(1)
        2.718281828459045

        >>> link_func = link_function('identity')
        >>> link_func(-2)
        -2
    """
    if link.lower() == "logistic":

        def def_link(shap_sum):
            return 1 / (1 + np.exp(-shap_sum))

    elif link.lower() == "logarithmic":

        def def_link(shap_sum):
            return np.exp(shap_sum)

    elif link.lower() == "identity":

        def def_link(shap_sum):
            return shap_sum

    else:
        raise ValueError("Invalid link defined. Must be 'logistic', 'logarithmic' or 'identity'.")
    return def_link


def calculate_row_sum(
    data: pd.DataFrame, mean_value: float, columns: List[str], link_function: Any
) -> Union[float, pd.Series]:
    """
    Calculates the row sum of specified columns in a Shapley values DataFrame and applies a link function to the result.

    Args:
        data (pd.DataFrame): The input DataFrame with shapley values.
        mean_value (float): The mean shapley value to be added to the row sum.
        columns (List[str]): The list of column names to be summed.
        link_function (callable): The link function to be applied to the result.

    Returns:
        Union[float, pd.Series]: The result of applying the link function to the row sum.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> mean_value = 10.0
        >>> columns = ['A', 'B']
        >>> link_function = lambda x: x ** 2
        >>> calculate_row_sum(data, mean_value, columns, link_function)
        225.0
    """
    return link_function(mean_value + data[columns].sum(axis=1))


def calculate_all_row_sum(data: pd.DataFrame, mean_value: float, link_function: Callable) -> Union[float, pd.Series]:
    """
    Calculates the row sum of **all columns** in a Shapley values DataFrame and applies a link function to the result.

    Args:
        data (pd.DataFrame): The input Shapley values DataFrame.
        mean_value (float): The mean shapley value to be added to the row sum.
        link_function (callable): The link function to be applied to the result.

    Returns:
        Union[float, pd.Series]: The result of applying the link function to the row sum.

    Examples:
        >>> import pandas as pd
        >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> mean_value = 10.0
        >>> link_function = lambda x: x ** 2
        >>> calculate_all_row_sum(data, mean_value, link_function)
        225.0
    """
    return link_function(mean_value + data.sum(axis=1))


def performance_fit(shaps_values, shap_expected_value, features, y_valid, link_function):
    aprox_preds = calculate_row_sum(shaps_values, features, shap_expected_value, link_function)
    return roc_auc_score(y_valid, aprox_preds)


def get_shap_tree_values(data: pd.DataFrame, model: Callable) -> Tuple[pd.DataFrame, float]:
    """
    Calculates the SHAP values and expected average shap value for a given dataset and model.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input dataset.
        model: The trained model object.

    Returns:
        tuple: A tuple containing the SHAP values and the Average shap value.

    Example:
        >>> # Imports
        >>> import numpy as np
        >>> from xgboost import XGBClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> # Imports SHAP Package
        >>> import shap
        >>>
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>>
        >>> # Split the dataset into train and test sets
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>>
        >>> # Train a model
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Calculate SHAP values and expected value
        >>> shap_values, expected_value = get_shap_tree_values(X_test, model)
    """
    shap_explainer = TreeExplainer(model)
    shap_valid = shap_explainer.shap_values(data)
    shap_expected_value = shap_explainer.expected_value

    return shap_valid, shap_expected_value


def generate_all_combinations(features: List[str]) -> List[Tuple[str]]:
    """
    Generates all possible combinations of the given features list.
    This will be used to test all possible combinations fo features to find the best combination.

    Args:
        features (List[str]): A list of features.

    Returns:
        List[Tuple[str]]: A list of tuples representing all possible combinations of the features.
    """
    all_combinations: List = []
    for feature_size in range(1, len(features) + 1):
        all_combinations.extend(combinations(features, feature_size))
    return all_combinations


def random_sort_shaps(
    shaps_values: pd.DataFrame,
    shap_expected_value: float,
    feature_name: str,
    y_target: Union[pd.Series, np.ndarray],
    link_function: Callable,
) -> float:
    """
    Randomly shuffles the values of a specific feature in the SHAP values DataFrame,
    calculates the row sum, and returns the ROC AUC score.

    Args:
        shaps_values (pd.DataFrame): The SHAP values DataFrame.
        shap_expected_value (float): The expected SHAP value.
        feature_name (str): The name of the feature to shuffle.
        y_target (Union[pd.Series, np.ndarray]): The target variable.
        link_function (callable): The link function to be applied to the row sum.

    Returns:
        float: The ROC AUC score.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import roc_auc_score
        >>>
        >>> # Generate synthetic data
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>>
        >>> # Train a logistic regression model
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Calculate SHAP values and expected value
        >>> shap_values, expected_value = get_shap_tree_values(X_test, model)
        >>>
        >>> # Calculate ROC AUC score with shuffled feature
        >>> roc_score = random_sort_shaps(shap_values, expected_value, 'feature_1', y_test, link_function='logistic')
        >>> print(roc_score)
    """
    shaps_values_shuffled = shaps_values.sample(frac=1)  # shuffle
    shaps_values_shuffled.reset_index(inplace=True, drop=True)

    new_shap_table = shaps_values.copy()
    new_shap_table.reset_index(inplace=True, drop=True)

    new_shap_table[feature_name] = shaps_values_shuffled[feature_name]
    approx_pred_valid = calculate_all_row_sum(new_shap_table, shap_expected_value, link_function)

    return roc_auc_score(y_target, approx_pred_valid)


def random_sort_shaps_column(  # pylint: disable=too-many-arguments
    shaps_values: pd.DataFrame,
    shap_mean_value: float,
    target_column: Union[pd.Series, np.ndarray],
    feature: str,
    link_function: Callable,
    original: bool = False,
) -> float:
    """
    Randomly shuffles the values of a specific feature in the SHAP values DataFrame,
    calculates the row sum, and returns the ROC AUC score.

    Args:
        shaps_values (pd.DataFrame): The SHAP values DataFrame.
        shap_mean_value (float): The mean SHAP value.
        target_column (Union[pd.Series, np.ndarray]): The target variable.
        feature (str): The name of the feature to shuffle.
        link_function (callable): The link function to be applied to the row sum.
        original (bool, optional): Whether to use the original feature values or shuffled values. Defaults to False.

    Returns:
        float: The ROC AUC score.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.metrics import roc_auc_score
        >>>
        >>> # Generate synthetic data
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>>
        >>> # Train a logistic regression model
        >>> model = LogisticRegression()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Calculate SHAP values and expected value
        >>> shap_values, expected_value = get_shap_tree_values(X_test, model)
        >>>
        >>> # Calculate ROC AUC score with shuffled feature
        >>> roc_score = random_sort_shaps_column(shap_values, expected_value, y_test, 'feature_1', link_function='logistic')
        >>> print(roc_score)
    """
    shaps_values_original = shaps_values.copy()
    shaps_values_original.reset_index(inplace=True, drop=True)

    shaps_values_shuffled = shaps_values.sample(frac=1)  # shuffle
    shaps_values_shuffled.reset_index(inplace=True, drop=True)

    # Calculate the average values of each column
    average_values = shaps_values.mean()
    new_shap_table = shaps_values.copy()
    new_shap_table.reset_index(inplace=True, drop=True)
    for feature_name in shaps_values.columns:
        new_shap_table[feature_name] = average_values[feature_name]

    if original:
        new_shap_table[feature] = shaps_values_original[feature]
    else:
        new_shap_table[feature] = shaps_values_shuffled[feature]

    approx_pred_valid = calculate_all_row_sum(new_shap_table, shap_mean_value, link_function)

    return roc_auc_score(target_column, approx_pred_valid)


def temp_plot_data(aprofs_obj, features: List[str]) -> pd.DataFrame:
    """
    Generate a temporary DataFrame for plotting purposes.

    Args:
        aprofs_obj (Aprofs Object): An instance of the Aprofs class.
        features (List[str]): A list of feature names.

    Returns:
        pd.DataFrame: The temporary DataFrame.

    Examples:
        >>> aprofs_obj = Aprofs Object(...)
        >>> features = ['feature_1', 'feature_2']
        >>> temp = temp_plot_data(aprofs_obj, features)
        >>> print(temp.head())
    """
    if not isinstance(features, list):
        features = [features]

    temp = pd.DataFrame(
        {
            "target": aprofs_obj.target_column,
        }
    )

    for feature in features:
        temp[feature] = aprofs_obj.current_data[feature].values
        temp[f"{feature}_shap"] = aprofs_obj.shap_mean + aprofs_obj.shap_values[feature].values
        temp[f"{feature}_shap_prob"] = 1 / (1 + np.exp(-temp[f"{feature}_shap"]))

    temp["shap_other"] = aprofs_obj.shap_mean + aprofs_obj.shap_values[
        [col for col in aprofs_obj.shap_values.columns if col not in features]
    ].sum(axis=1)
    temp["shap_prob_other"] = 1 / (1 + np.exp(-temp["shap_other"]))
    temp["shap_model"] = aprofs_obj.shap_mean + aprofs_obj.shap_values.sum(axis=1)
    temp["shap_prob_model"] = 1 / (1 + np.exp(-temp["shap_model"]))

    return temp


def plot_data(  # pylint: disable=too-many-arguments
    temp: pd.DataFrame,
    main_feature: str,
    other_features: Optional[Union[str, List[str]]] = None,
    nbins: int = 20,
    type_bins: str = "qcut",
    type_plot: str = "prob",
) -> None:
    """
    Plot data based on the provided DataFrame and features.

    Args:
        temp (pd.DataFrame): The DataFrame containing the data.
        main_feature (str): The main feature to plot.
        other_features (Optional[Union[str, List[str]]], optional): Other features to include in the plot. Defaults to None.
        nbins (int, optional): The number of bins. Defaults to 20.
        type_bins (str, optional): The type of binning. Defaults to "qcut".
        type_plot (str, optional): The type of plot. Defaults to "prob".

    Returns:
        None

    Examples:
        >>> temp = pd.DataFrame(...)
        >>> plot_data(temp, "main_feature", other_features=["feature_1", "feature_2"], nbins=10, type_bins="cut", type_plot="raw")
    """
    if other_features is None:
        other_features = []
    if not isinstance(other_features, list):
        other_features = [other_features]
    features = []
    features.append(main_feature)
    features.extend(other_features)

    if temp[main_feature].unique().shape[0] < 25:
        temp["bins"] = temp[main_feature].astype(str)
    elif type_bins == "cut":
        temp["bins"] = pd.cut(temp[main_feature], bins=nbins)
    elif type_bins == "qcut":
        temp["bins"] = pd.qcut(temp[main_feature], q=nbins)
    else:
        print("Invalid type_bins value")

    # Calculate the means for each bin
    means = temp.groupby("bins", observed=True)["target"].mean()

    means_shap = {}
    if type_plot == "raw":
        for feature in features:
            means_shap[feature] = temp.groupby("bins", observed=True)[f"{feature}_shap"].mean()
        means_shap_others = temp.groupby("bins", observed=True)["shap_other"].mean()
        means_shap_model = temp.groupby("bins", observed=True)["shap_model"].mean()
    else:
        for feature in features:
            means_shap[feature] = temp.groupby("bins", observed=True)[f"{feature}_shap_prob"].mean()
        means_shap_others = temp.groupby("bins", observed=True)["shap_prob_other"].mean()
        means_shap_model = temp.groupby("bins", observed=True)["shap_prob_model"].mean()

    # Calculate the counts for each bin
    counts = temp["bins"].value_counts(normalize=True).sort_index()

    # Create a figure
    fig = go.Figure()

    # Add bar plot for counts on the primary y-axis
    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts, name="Data", yaxis="y", marker_color="lightgray"))

    # Add line plots on the secondary y-axis
    fig.add_trace(go.Scatter(x=means.index.astype(str), y=means, mode="lines", name="Observed", yaxis="y2"))

    for feature in features:
        fig.add_trace(
            go.Scatter(
                x=means_shap[feature].index.astype(str),
                y=means_shap[feature],
                mode="lines",
                name=f"{feature} shap Mean",
                yaxis="y2",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=means_shap_others.index.astype(str), y=means_shap_others, mode="lines", name="Others shaps", yaxis="y2"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=means_shap_model.index.astype(str), y=means_shap_model, mode="lines", name="Model shaps", yaxis="y2"
        )
    )

    # Update layout to include a secondary y-axis
    fig.update_layout(
        yaxis={"title": "Counts", "side": "left", "tickformat": ".0%"},
        yaxis2={"title": "Avg.", "side": "right", "overlaying": "y"},
    )

    fig.show()
