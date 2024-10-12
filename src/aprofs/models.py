"""This module implements the models class.
this wasy we can extend this class to implement new models
to calculate the use with the aprofs class"""

import abc
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
)


# Model Interface
class LinkModels(metaclass=abc.ABCMeta):
    """This class implements the interface for the link models
    to be used in the aprofs class

    Functionality that needs ot be implemented:
        - performance_fit: calculate the performance of the model
        - link_calculate: calculate the link function
        - inv_link_calculate: calculate the inverse link function

    """

    def __init__(self, type_model: str, type_link: str, perform: str) -> None:
        self.type_model = type_model
        self.type_link = type_link
        self.perform = perform

    @abc.abstractmethod
    def performance_fit(self, target: Union[np.ndarray, pd.Series], prediction: Union[np.ndarray, pd.Series]) -> float:
        """
        Calculate the performance of the model.

        Args:
            target (np.ndarray): The true target values.
            prediction (np.ndarray): The predicted values.

        Returns:
            float: The performance metric.
        """

    @abc.abstractmethod
    def link_calculate(
        self, inv_prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray]:
        """
        Calculate the link function.

        Args:
            inv_prediction (Union[int, float, np.ndarray]): The input value(s).

        Returns:
            Union[int, float, np.ndarray]: The transformed value(s).
        """

    @abc.abstractmethod
    def inv_link_calculate(
        self, prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        """
        Calculate the inverse link function.

        Args:
            prediction (Union[int, float, np.ndarray]): The input value(s).

        Returns:
            Union[int, float, np.ndarray]: The transformed value(s).
        """


class RegressionLogLink(LinkModels):
    """This class implements the interface for regression with logarithmic link"""

    def __init__(self) -> None:
        super().__init__(type_model="regression", type_link="logarithmic", perform="minimize")

    def performance_fit(self, target: Union[np.ndarray, pd.Series], prediction: Union[np.ndarray, pd.Series]) -> float:
        return np.sqrt(mean_squared_error(target, prediction))

    def link_calculate(
        self, inv_prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        if not isinstance(inv_prediction, (int, float, np.ndarray, pd.Series)):
            raise ValueError("Invalid input type for link_calculate")
        return np.log(inv_prediction)

    def inv_link_calculate(
        self, prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        return np.exp(prediction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}() with type model {self.type_model} and type link {self.type_link}"


class RegressionIdentityLink(LinkModels):
    """This class implements the interface for regression with identity link"""

    def __init__(self) -> None:
        super().__init__(type_model="regression", type_link="identity", perform="minimize")

    def performance_fit(self, target: Union[np.ndarray, pd.Series], prediction: Union[np.ndarray, pd.Series]) -> float:
        return np.sqrt(mean_squared_error(target, prediction))

    def link_calculate(
        self, inv_prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        return inv_prediction

    def inv_link_calculate(
        self, prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        return prediction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}() with type model {self.type_model} and type link {self.type_link}"


class ClassificationLogisticLink(LinkModels):
    """This class implements the interface for classification with logistic link"""

    def __init__(self) -> None:
        super().__init__(type_model="classification", type_link="logistic", perform="maximize")

    def performance_fit(self, target: Union[np.ndarray, pd.Series], prediction: Union[np.ndarray, pd.Series]) -> float:
        return roc_auc_score(target, prediction)

    def link_calculate(
        self, inv_prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        if not isinstance(inv_prediction, (int, float, np.ndarray, pd.Series)):
            raise ValueError("Invalid input type for link_calculate")
        return 1 / (1 + np.exp(-inv_prediction))

    def inv_link_calculate(
        self, prediction: Union[int, float, np.ndarray, pd.Series]
    ) -> Union[int, float, np.ndarray, pd.Series]:
        return np.log(prediction / (1 - prediction))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}() with type model {self.type_model} and type link {self.type_link}"
