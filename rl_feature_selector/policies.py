import random
from typing import List
from abc import ABC, abstractmethod

import numpy as np


class Policy(ABC):
    @abstractmethod
    def select_action(self, state: List[int]) -> int:
        pass


class RandomPolicy(Policy):
    def select_action(self, state: List[int]) -> int:
        """
        Randomly select an action (feature) to add to the current state.

        :param state: The current state represented as a list of binary values.
        :return: The index of the feature to be added.
        """
        # Find indices of features not already in the state
        available_features = [
            i for i, feature in enumerate(state) if feature == 0
        ]

        # Randomly select an index from the available features
        return random.choice(available_features)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def select_action(self, state: List[int], aor: np.ndarray) -> int:
        """
        Select an action based on epsilon-greedy policy.

        :param state: The current state represented as a list of binary values.
        :return: The index of the feature to be added.
        """
        if random.random() < self._epsilon:
            available_features = [
                i for i, feature in enumerate(state) if feature == 0
            ]
            return random.choice(available_features)
        else:
            # Get AOR values only for available (not yet selected) features
            available_aor = [
                (i, aor[1, i])
                for i, feature in enumerate(state)
                if feature == 0
            ]

            # Select the action (feature) with the highest AOR value
            return max(available_aor, key=lambda x: x[1])[0]
