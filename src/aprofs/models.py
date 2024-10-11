"""This module implements the models class.
this wasy we can extend this class to implement new models
to calculate the use with the aprofs class"""

import abc

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    roc_auc_score,
)


# Model Interface
class LinkModels(metaclass=abc.ABCMeta):
    """This class implements the interface for the link models

    Args:
        metaclass (_type_, optional): _description_. Defaults to abc.ABCMeta.
    """

    @abc.abstractmethod
    def performance_fit(self, target, prediction):
        pass

    @abc.abstractmethod
    def link_calculate(self, inv_prediction):
        pass

    @abc.abstractmethod
    def inv_link_calculate(self, prediction):
        pass


class RegressionLogLink(LinkModels):
    """This class implements the link model for regression with logarithmic link"""

    def __init__(self) -> None:
        super().__init__()
        self.type_model = "regression"
        self.type_link = "logarithmic"
        self.perform = "minimize"

    def performance_fit(self, target, prediction):
        return np.sqrt(mean_squared_error(target, prediction))

    def link_calculate(self, inv_prediction):
        return np.log(inv_prediction)

    def inv_link_calculate(self, prediction):
        return np.exp(prediction)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}() with type model {self.type_model} and type link {self.type_link}"


class ClassificationLogisticLink(LinkModels):
    """This class implements the link model for classification with logistic link

    Args:
        LinkModels (_type_): _description_
    """

    def __init__(self) -> None:
        super().__init__()
        self.type_model = "classification"
        self.type_link = "logistic"
        self.perform = "maximize"

    def performance_fit(self, target, prediction):
        return roc_auc_score(target, prediction)

    def link_calculate(self, inv_prediction):
        return 1 / (1 + np.exp(-inv_prediction))

    def inv_link_calculate(self, prediction):
        return np.log(prediction / (1 - prediction))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}() with type model {self.type_model} and type link {self.type_link}"
