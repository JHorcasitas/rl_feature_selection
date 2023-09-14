from typing import List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC


class CrossValidationMixin:

    def generate_stratified_kfolds(self, Y: np.ndarray, num_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """   
        This method creates stratified splits of the data, ensuring that each fold is made 
        by preserving the percentage of samples for each class. 
    
        :param Y: A one-dimensional numpy array of target labels.
        :param num_splits: Number of splits / folds to generate.
        :return: A list of tuples where each tuple contains training and test indices.
        """
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True)
        return [(train_index, test_index) for train_index, test_index in skf.split(np.zeros(len(Y)), Y)]


class ModelTrainer(ABC):
    # def train_and_evaluate(self, X: pd.DataFrame, Y: pd.Series, train_indices: List[np.ndarray], test_indices: List[np.ndarray]) -> float:
    @abstractmethod
    def train_and_evaluate(self, X: pd.DataFrame, Y: pd.Series) -> float:
        """Train the model and evaluate using k-fold cross-validation, return average AUPRC."""
        pass


class GaussianSVMTrainer(ModelTrainer, CrossValidationMixin):
    def __init__(self, C: float = 1.0, kernel: str = 'rbf'):
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def _train_single_fold(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on a single fold."""
        self.model.fit(X_train, y_train)

    def _evaluate_single_fold(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model on a single fold and return AUPRC."""
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        return average_precision_score(y_test, y_pred_prob)

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
    ) -> float:
        total_auprc = 0.0
        for train_indices, test_indices in self.generate_stratified_kfolds(Y, num_splits=2):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
            self._train_single_fold(X_train, Y_train)
            auprc = self._evaluate_single_fold(X_test, Y_test)
            total_auprc += auprc
        return total_auprc / len(train_indices)


class LinearSVMTrainer(ModelTrainer, CrossValidationMixin):
    def __init__(self, C: float = 1.0):
        # Initialize the LinearSVC model
        svm = LinearSVC(C=C, class_weight="balanced", dual="auto")
        # Calibrate the model to provide probability estimates
        self.model = CalibratedClassifierCV(svm)

    def _train_single_fold(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model on a single fold."""
        self.model.fit(X_train, y_train)

    def _evaluate_single_fold(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate the model on a single fold and return AUPRC."""
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]
        return average_precision_score(y_test, y_pred_prob)

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        num_splits: int = 2
    ) -> float:
        np.random.seed(0)
        total_auprc = 0.0
        for train_indices, test_indices in self.generate_stratified_kfolds(Y, num_splits):
            X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
            Y_train, Y_test = Y.iloc[train_indices], Y.iloc[test_indices]
            self._train_single_fold(X_train, Y_train)
            auprc = self._evaluate_single_fold(X_test, Y_test)
            total_auprc += auprc
        return total_auprc / num_splits
