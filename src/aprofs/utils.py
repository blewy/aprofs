"""Utility functions for the package.
this module contains utility functions that are used in the package.
the core functions are used to calculate the SHAP values and expected average SHAP value for a given dataset and model.
"""

from itertools import combinations
from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shap import (
    Explainer,
    TreeExplainer,
)
from sklearn.metrics import roc_auc_score

from .models import LinkModels


def calculate_row_sum(
    data: pd.DataFrame, mean_value: float, columns: List[str], link_model: LinkModels
) -> Union[float, pd.Series]:
    """
    Calculates the row sum of specified columns in a Shapley values DataFrame and applies a link function to the result.

    Args:
        data (pd.DataFrame): The input DataFrame with shapley values.
        mean_value (float): The mean shapley value to be added to the row sum.
        columns (List[str]): The list of column names to be summed.
        link_model (aprofs model object): An object that allows to calculate the performance of the model.

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
    return link_model.link_calculate(mean_value + data[columns].sum(axis=1))


def calculate_all_row_sum(data: pd.DataFrame, mean_value: float, link_model: LinkModels) -> Union[float, pd.Series]:
    """
    Calculates the row sum of **all columns** in a Shapley values DataFrame and applies a link function to the result.

    Args:
        data (pd.DataFrame): The input Shapley values DataFrame.
        mean_value (float): The mean shapley value to be added to the row sum.
        link_model (aprofs model object): An object that allows to calculate the performance of the model.

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
    return link_model.link_calculate(mean_value + data.sum(axis=1))


def performance_fit(
    shaps_values: pd.DataFrame,
    shap_expected_value: float,
    features: List[str],
    y_valid: pd.Series,
    link_model: LinkModels,
) -> float:
    aprox_preds = calculate_row_sum(shaps_values, shap_expected_value, features, link_model)
    return link_model.performance_fit(y_valid, aprox_preds)


def best_feature(  # pylint: disable=too-many-arguments
    shaps_values: pd.DataFrame,
    shap_expected_values: float,
    link_model: LinkModels,
    y_target: pd.Series,
    current_list: List[str],
    candidate_list: List[str],
) -> Tuple[str, float]:
    """
    Return the best feature to add to the current list based on the highest AUC score.

    Args:
        shaps_values (DataFrame): A DataFrame containing SHAP values for each feature.
        shap_expected_values (Series): A Series containing the expected SHAP values.
        link_model (aprofs model object): An object that allows to calculate the performance of the model.
        y_target (Series): The target variable for the AUC score calculation.
        current_list (list): The current list of features.
        candidate_list (list): The list of candidate features to consider adding.

    Returns:
        tuple: A tuple containing the best feature to add (str) and the corresponding best AUC score (float).

    Raises:
        ValueError: If `candidate_list` is empty.
    """

    if candidate_list == [] or candidate_list is None:
        raise ValueError("The candidate list cannot be empty.")

    best_feature: str = None
    best_auc: float = 0
    for feature in candidate_list:
        current_list.append(feature)
        aprox_preds = calculate_row_sum(shaps_values, shap_expected_values, current_list, link_model)
        auc = roc_auc_score(y_target, aprox_preds)
        if auc > best_auc:
            best_auc = auc
            best_feature = feature
        current_list.remove(feature)
    return best_feature, best_auc


def get_shap_values(data: pd.DataFrame, model: Callable, type_model="tree") -> Tuple[pd.DataFrame, float]:
    """
    Calculates the SHAP values and expected average shap value for a given dataset and model.

    Args:
        data (numpy.ndarray or pandas.DataFrame): The input dataset.
        model: The trained model object.
        type_model (str): type of model: tree based or other. If "tree" then TreeExplainer will be use, otherwise a general explainer from the SHAP package is used. Defaults to 'tree'.

    Returns:
        tuple: A tuple containing the SHAP values and the Average shap value.

    Examples:
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
    if type_model == "tree":
        shap_explainer = TreeExplainer(model)
        shap_valid = shap_explainer.shap_values(data)
        shap_expected_value = shap_explainer.expected_value
    else:
        shap_explainer = Explainer(model)
        shap_valid = shap_explainer.shap_values(data)
        shap_expected_value = shap_explainer.expected_value

    if isinstance(shap_valid, list):
        shap_valid = np.concatenate(shap_valid, axis=1)

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
    link_model: LinkModels,
) -> float:
    """
    Randomly shuffles the values of a specific feature in the SHAP values DataFrame,
    calculates the row sum, and returns the ROC AUC score.

    Args:
        shaps_values (pd.DataFrame): The SHAP values DataFrame.
        shap_expected_value (float): The expected SHAP value.
        feature_name (str): The name of the feature to shuffle.
        y_target (Union[pd.Series, np.ndarray]): The target variable.
        link_model (aprofs model object): An object that allows to calculate the performance of the model.

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
    approx_pred_valid = calculate_all_row_sum(new_shap_table, shap_expected_value, link_model)

    return link_model.performance_fit(y_target, approx_pred_valid)


def random_sort_shaps_column(  # pylint: disable=too-many-arguments
    shaps_values: pd.DataFrame,
    shap_mean_value: float,
    target_column: Union[pd.Series, np.ndarray],
    feature: str,
    link_model: LinkModels,
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
        link_model (aprofs model object): An object that allows to calculate the performance of the model.
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

    approx_pred_valid = calculate_all_row_sum(new_shap_table, shap_mean_value, link_model)

    return link_model.performance_fit(target_column, approx_pred_valid)


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
    fig.update_xaxes(title_text=feature)
    fig.show()


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


def plot_data_compare(  # pylint: disable=too-many-arguments
    temp: pd.DataFrame,
    feature: str,
    nbins: int = 20,
    type_bins: str = "qcut",
    type_plot: str = "prob",
) -> None:
    """
    Plot data based on the provided DataFrame and feature in a way to compare a specific shap.

    Args:
        temp (pd.DataFrame): The DataFrame containing the data.
        feature (str): The main feature to plot.
        nbins (int, optional): The number of bins. Defaults to 20.
        type_bins (str, optional): The type of binning. Defaults to "qcut".
        type_plot (str, optional): The type of plot. Defaults to "prob".

    Returns:
        None

    Examples:
        >>> temp = pd.DataFrame(...)
        >>> plot_data(temp, "feature_name", nbins=10, type_bins="cut", type_plot="raw")
    """

    if temp[feature].unique().shape[0] < 25:
        temp["bins"] = temp[feature].astype(str)
    elif type_bins == "cut":
        temp["bins"] = pd.cut(temp[feature], bins=nbins)
    elif type_bins == "qcut":
        temp["bins"] = pd.qcut(temp[feature], q=nbins)
    else:
        print("Invalid type_bins value")

    # Calculate the means for each bin
    means = temp.groupby("bins", observed=True)["target"].mean()

    means_shap = {}
    if type_plot == "raw":
        means_shap[feature] = temp.groupby("bins", observed=True)[f"{feature}_shap"].mean()
        means_shap[f"{feature}_shap"] = temp.groupby("bins", observed=True)[f"{feature}_shap"].mean()
        means_shap[f"{feature}_shap_compare"] = temp.groupby("bins", observed=True)[f"{feature}_shap_compare"].mean()
        means_shap_model = temp.groupby("bins", observed=True)["shap_model"].mean()
    else:
        means_shap[feature] = temp.groupby("bins", observed=True)[f"{feature}_shap_prob"].mean()
        means_shap[f"{feature}_compare"] = temp.groupby("bins", observed=True)[f"{feature}_shap_prob_compare"].mean()
        means_shap_model = temp.groupby("bins", observed=True)["shap_prob_model"].mean()

    # Calculate the counts for each bin
    counts = temp["bins"].value_counts(normalize=True).sort_index()

    # Create a figure
    fig = go.Figure()

    # Add bar plot for counts on the primary y-axis
    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts, name="Data", yaxis="y", marker_color="lightgray"))

    # Add line plots on the secondary y-axis
    fig.add_trace(go.Scatter(x=means.index.astype(str), y=means, mode="lines", name="Observed", yaxis="y2"))

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
            x=means_shap[f"{feature}_compare"].index.astype(str),
            y=means_shap[f"{feature}_compare"],
            mode="lines",
            name=f"{feature} shap Mean compare",
            yaxis="y2",
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
    fig.update_xaxes(title_text=feature)
    fig.show()


def temp_plot_compare_data(aprofs_obj_self, aprofs_obj, feature: str) -> pd.DataFrame:
    """
    Generate a temporary DataFrame for plotting purposes.

    Args:
        aprofs_obj_self (Aprofs Object): An instance of the Aprofs class.
        aprofs_obj (Aprofs Object): An instance of the Aprofs class.
        feature (str): feature to compare.

    Returns:
        pd.DataFrame: The temporary DataFrame.

    Examples:
        >>> aprofs_obj = Aprofs Object(...)
        >>> aprofs_obj_2_compare = Aprofs Object(...)
        >>> features = 'feature_1'
        >>> temp = temp_plot_data(aprofs_obj,aprofs_obj_2_compare, feature)
        >>> print(temp.head())
    """

    temp = pd.DataFrame(
        {
            "target": aprofs_obj_self.target_column,
        }
    )

    # self data
    temp[feature] = aprofs_obj_self.current_data[feature].values
    temp[f"{feature}_shap"] = aprofs_obj_self.shap_mean + aprofs_obj_self.shap_values[feature].values
    temp[f"{feature}_shap_prob"] = 1 / (1 + np.exp(-temp[f"{feature}_shap"]))

    # compare data
    temp[f"{feature}_shap_compare"] = aprofs_obj.shap_mean + aprofs_obj.shap_values[feature].values
    temp[f"{feature}_shap_prob_compare"] = 1 / (1 + np.exp(-temp[f"{feature}_shap_compare"]))

    # model probabilities data
    temp["shap_model"] = aprofs_obj.shap_mean + aprofs_obj.shap_values.sum(axis=1)
    temp["shap_prob_model"] = 1 / (1 + np.exp(-temp["shap_model"]))

    return temp


def plot_data_neutral(  # pylint: disable=too-many-arguments
    data: pd.DataFrame,
    feature: str,
    nbins: int = 20,
    type_bins: str = "qcut",
    type_plot: str = "prob",
) -> None:
    """
    Plot data based on the provided neutralized DataFrame and features.

    Args:
        data (pd.DataFrame): The DataFrame containing the neutralize shap data.
        feature (str): The main feature to plot on the x-axis.
        nbins (int, optional): The number of bins. Defaults to 20.
        type_bins (str, optional): The type of binning. Defaults to "qcut".
        type_plot (str, optional): The type of plot. Defaults to "prob".

    Returns:
        None

    Examples:
        >>> temp = pd.DataFrame(...)
        >>> plot_data_neutral(temp, "main_feature", other_features=["feature_1", "feature_2"], nbins=10, type_bins="cut", type_plot="raw")
    """

    if data[feature].unique().shape[0] < 25:
        data["bins"] = data[feature].astype(str)
    elif type_bins == "cut":
        data["bins"] = pd.cut(data[feature], bins=nbins)
    elif type_bins == "qcut":
        data["bins"] = pd.qcut(data[feature], q=nbins)
    else:
        print("Invalid type_bins value")

    # Calculate the means for each bin
    means = data.groupby("bins", observed=True)["target"].mean()

    if type_plot == "raw":
        means_shap_others = data.groupby("bins", observed=True)["shap_other"].mean()
        means_shap_model = data.groupby("bins", observed=True)["shap_model"].mean()
    else:
        means_shap_others = data.groupby("bins", observed=True)["shap_prob_other"].mean()
        means_shap_model = data.groupby("bins", observed=True)["shap_prob_model"].mean()

    # Calculate the counts for each bin
    counts = data["bins"].value_counts(normalize=True).sort_index()

    # Create a figure
    fig = go.Figure()

    # Add bar plot for counts on the primary y-axis
    fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts, name="Data", yaxis="y", marker_color="lightgray"))

    # Add line plots on the secondary y-axis
    fig.add_trace(go.Scatter(x=means.index.astype(str), y=means, mode="lines", name="Observed", yaxis="y2"))

    fig.add_trace(
        go.Scatter(
            x=means_shap_others.index.astype(str),
            y=means_shap_others,
            mode="lines",
            name="Neutralized shaps",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=means_shap_model.index.astype(str),
            y=means_shap_model,
            mode="lines",
            name="Original Model shaps",
            yaxis="y2",
        )
    )

    # Update layout to include a secondary y-axis
    fig.update_layout(
        yaxis={"title": "Counts", "side": "left", "tickformat": ".0%"},
        yaxis2={"title": "Avg.", "side": "right", "overlaying": "y"},
    )
    # Add title to x-axis
    fig.update_xaxes(title_text=feature)

    fig.show()


def temp_neutral_plot_data(aprofs_obj, features: List[str]) -> pd.DataFrame:
    """
    Generate a temporary DataFrame for plotting purposes.

    Args:
        aprofs_obj (Aprofs Object): An instance of the Aprofs class.
        features (List[str]): A list of feature names that will be neutralized. The shapley values for this will be just the average values. This way the break the segmentation of the feature, maintaining the global effect of all the others.

    Returns:
        pd.DataFrame: The temporary DataFrame.

    Examples:
        >>> aprofs_obj = Aprofs Object(...)
        >>> features = ['feature_1', 'feature_2']
        >>> temp = temp_neutral_plot_data(aprofs_obj, features)
        >>> print(temp.head())
    """
    if not isinstance(features, list):
        features = [features]

    temp = pd.DataFrame(
        {
            "target": aprofs_obj.target_column,
        }
    )

    for feat in features:
        temp[feat] = aprofs_obj.current_data[feat].values  # adding features to data

    temp["shap_other"] = (
        aprofs_obj.shap_mean
        + aprofs_obj.shap_values[[col for col in aprofs_obj.shap_values.columns if col not in features]].sum(axis=1)
        + aprofs_obj.shap_values[features]
        .sum(axis=1)
        .mean()  # sums the columns of the features and calculate the average value
    )
    temp["shap_prob_other"] = 1 / (1 + np.exp(-temp["shap_other"]))
    temp["shap_model"] = aprofs_obj.shap_mean + aprofs_obj.shap_values.sum(axis=1)
    temp["shap_prob_model"] = 1 / (1 + np.exp(-temp["shap_model"]))

    return temp
