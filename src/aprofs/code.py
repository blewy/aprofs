"""
Core code development of the Aprofs class.

This class is used to calculate the SHAP values of a model and
evaluate the performance of the features based on the SHAP values. The class also provides a method to
visualize the marginal effect of a feature on the target variable. The class is used to perform feature
selection based on the SHAP values and to calculate the p-values of the SHAP values of the features.
The class is also used to calculate the performance of the model based on the SHAP values of the features.
"""

# Importing the required libraries
from typing import (
    Any,
    List,
)

import pandas as pd
from tqdm import tqdm

from aprofs import utils

from .models import (
    ClassificationLogisticLink,
    LinkModels,
)


class Aprofs:
    """
    Aprofs Class

    A class for analyzing SHAP values using approximate predictions.
    --------------------------------------------------------------


    Attributes:
        current_data (pd.DataFrame): The current data.
        target_column (Series): The target column.
        link (function): The link function.
        link_srt (str): The string representation of the link function.
        shap_mean (float): The mean SHAP value. None if SHAP values have not been calculated.
        shap_values (DataFrame): The SHAP values. None if SHAP values have not been calculated.

    """

    def __init__(self, current_data, target_column, link_model: LinkModels):
        self.current_data = current_data
        self.target_column = target_column
        self.link_model = ClassificationLogisticLink() if link_model is None else link_model
        self.shap_mean: float = None
        self.shap_values: pd.DataFrame = None

    def __repr__(self):
        return (
            f"Aprofs(current_data shape ={self.current_data.shape}, target_column ={self.target_column.unique()}"
            + (
                f", shap_mean={self.shap_mean}, shap_values.shape={self.shap_values.shape}"
                if self.shap_mean is not None
                else "\n  Shapley values have not been calculated!"
            )
        )

    def calculate_shaps(self, model: Any, type_model="tree") -> None:
        """
        Calculate the SHAP values for the given model.

        Parameters:
            model (Any): The trained model for which to calculate the SHAP values.
            type_model (str): type of model: tree based or other. If "tree" then TreeExplainer will be use, otherwise a general explainer from the SHAP package is used. Defaults to 'tree'.


        Returns:
            None
        """
        shap_values, shap_mean = utils.get_shap_values(self.current_data, model, type_model)
        self.shap_values = pd.DataFrame(shap_values, index=self.current_data.index, columns=self.current_data.columns)
        self.shap_mean = shap_mean

    def get_feature_performance(self, features: List[str]) -> float:
        """
        Calculate the performance of the features based on the SHAP values.

        Parameters:
            features (List[str]): The list of features for which to calculate the performance.

        Returns:
            float: The performance of the features based on the SHAP values.

        Raises:
            ValueError: If an any feature is missing in the SHAP values.
        """
        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")
        return self.link_model.performance_fit(
            self.target_column, utils.calculate_row_sum(self.shap_values, self.shap_mean, features, self.link_model)
        )

    def brute_force_selection(self, features: List[str]) -> List[str]:
        """
        Perform brute force feature selection by evaluating the performance of all possible combinations of features.

        Parameters:
            features (List[str]): The list of features to consider for feature selection.

        Returns:
            List[str]: The best list of features with the highest performance.

        Raises:
            ValueError: If an any feature is missing in the SHAP values.
        """
        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")

        best_performance = 0.0
        best_list = []
        all_combinations = list(utils.generate_all_combinations(features))
        for comb in tqdm(all_combinations, desc=f"Processing {len(all_combinations)} combinations"):
            current_performance = self.get_feature_performance(list(comb))
            if current_performance > best_performance:
                best_performance = current_performance
                best_list = comb
        print(f"the best list is {best_list} with performance {best_performance}")
        return list(best_list)

    def gready_forward_selection(self, features: List[str], greediness: float = 0.001) -> List[str]:
        """
        Perform gready forward feature selection by evaluating the performance of all possible combinations of features.

        Parameters:
            features (List[str]): The list of features to consider for feature selection.
            greediness (float): The greediness factor, how much better needs to be the performance to add the feature. Default is 0.001.
        Returns:
            List[str]: The best list of features with the highest performance.

        Raises:
            ValueError: If an any feature is missing in the SHAP values.
        """
        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")

        best_list: List = []
        candidate_list: List[str] = features.copy()
        aproximate_performance: List[float] = []
        best_performance = 0.0
        while len(candidate_list) > 0:
            best_feature_, best_performance_ = utils.best_feature(
                self.shap_values, self.shap_mean, self.link_model, self.target_column, best_list, candidate_list
            )
            candidate_list.remove(best_feature_)

            if best_performance > best_performance_ * (1 + greediness):
                print(f"The feature {best_feature_} wont be added")
            else:
                best_performance = best_performance_
                best_list.append(best_feature_)
                print(f"the best feature to add is {best_feature_} with performance {best_performance_}")
                aproximate_performance.append(best_performance_)

        return best_list

    def get_shap_p_value(self, features: List[str], suffle_size: int = 500) -> pd.DataFrame:
        """
        Calculate the p-values of the SHAP values of the features.

        Parameters:
            features (List[str]): The list of features for which to calculate the p-values.
            suffle_size (int): The number of shuffling iterations to perform. Default is 500.

        Returns:
            pd.DataFrame: A DataFrame containing the features and their corresponding p-values.

        Raises:
            ValueError: If an any feature is missing in the SHAP values.
        """
        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")

        p_values = []
        performance_threshold = self.get_feature_performance(self.shap_values.columns)
        for feature in tqdm(features):
            samples = [
                utils.random_sort_shaps(self.shap_values, self.shap_mean, feature, self.target_column, self.link_model)
                for _ in range(suffle_size)
            ]
            count = sum(sample > performance_threshold for sample in samples)
            p_values.append(count / suffle_size)

        return pd.DataFrame({"Feature": features, "p-value_shap": p_values})

    def visualize_feature(  # pylint: disable=too-many-arguments
        self,
        main_feature: str,
        other_features: List[str] = None,
        nbins: int = 20,
        type_bins: str = "qcut",
        type_plot: str = "prob",
    ) -> None:
        """
        Visualize the marginal effect of a feature on the target variable.

        Parameters:
            main_feature (str): The main feature for which to visualize the marginal effect.
            other_features (List[str]): The list of other features to include in the visualization. Default is None.
            nbins (int): The number of bins to use for the visualization. Default is 20.
            type_bins (str): The type of binning to use. Default is "qcut".
            type_plot (str): The type of plot to generate. Default is "prob".

        Returns:
            None

        Raises:
            ValueError: If an any feature is missing in the SHAP values dataframe.
        """
        # generate data to plot marginal effect shapley values
        if other_features is None:
            other_features = []
        features = []
        features.append(main_feature)
        features.extend(other_features)

        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")

        temp_data = utils.temp_plot_data(self, features)
        # call plotting function
        utils.plot_data(
            temp_data,
            main_feature,
            other_features=other_features,
            nbins=nbins,
            type_bins=type_bins,
            type_plot=type_plot,
        )

    def compare_feature(  # pylint: disable=too-many-arguments
        self,
        other,
        feature: str,
        nbins: int = 20,
        type_bins: str = "qcut",
        type_plot: str = "prob",
    ) -> None:
        """
        Visualize the marginal effect of a feature on the target variable.

        Parameters:
            feature (str): The main feature for which to visualize the marginal effect.
            nbins (int): The number of bins to use for the visualization. Default is 20.
            type_bins (str): The type of binning to use. Default is "qcut".
            type_plot (str): The type of plot to generate. Default is "prob".

        Returns:
            None

        Raises:
            ValueError: If an any feature is missing in the SHAP values.
        """

        if not isinstance(other, Aprofs):
            raise ValueError("Can only compare with another Aprofs object")

        if feature not in self.shap_values.columns:
            raise ValueError(f"The following feature are missing in the SHAP values: {feature}")

        temp_data = utils.temp_plot_compare_data(self, other, feature)
        # call plotting function
        utils.plot_data_compare(
            temp_data,
            feature,
            nbins=nbins,
            type_bins=type_bins,
            type_plot=type_plot,
        )

    def visualize_neutralized_feature(  # pylint: disable=too-many-arguments
        self,
        main_feature: str,
        neutralize_features: List[str] = None,
        nbins: int = 20,
        type_bins: str = "qcut",
        type_plot: str = "prob",
    ) -> None:
        """
        Visualize the marginal effect of a feature on the target variable after neutralizing the effect of other features.

        Parameters:
            main_feature (str): The main feature for which to visualize the marginal effect.
            neutralize_features (List[str]): The list of other features to be neutralized.
            nbins (int): The number of bins to use for the visualization. Default is 20.
            type_bins (str): The type of binning to use. Default is "qcut".
            type_plot (str): The type of plot to generate. Default is "prob".

        Returns:
            None

        Raises:
            ValueError: If an any feature is missing in the SHAP values dataframe.
        """
        # generate data to plot marginal effect shapley values
        if neutralize_features is None:
            neutralize_features = []
        features = []
        if not isinstance(neutralize_features, list):
            neutralize_features = [neutralize_features]

        features.append(main_feature)
        features.extend(neutralize_features)
        features = list(set(features))  # remove duplicates

        missing_features = [feature for feature in features if feature not in self.shap_values.columns]
        if missing_features:
            raise ValueError(f"The following features are missing in the SHAP values: {missing_features}")

        temp_data = utils.temp_neutral_plot_data(self, neutralize_features)
        temp_data[main_feature] = self.current_data[main_feature]
        # call plotting function
        utils.plot_data_neutral(
            temp_data,
            main_feature,
            nbins=nbins,
            type_bins=type_bins,
            type_plot=type_plot,
        )
