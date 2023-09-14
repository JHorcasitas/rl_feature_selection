import os
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

from rl_feature_selector import policies, train, state_space


logger = logging.getLogger("feature_selector")
logger.addHandler(logging.FileHandler(os.path.join("logs", "feature_selector_2.log"), mode="a"))
logger.setLevel(logging.DEBUG)
logger.propagate = False


class FeatureSelector:
    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        num_episodes: int,
        no_previous_visit_policy: policies.Policy,
        previous_visit_policy: policies.Policy,
        model_trainer: train.ModelTrainer,
        alpha: float = 0.1,
    ):
        self._X = X
        self._Y = Y
        self._alpha = alpha
        self._number_of_features = X.shape[1]
        self._num_episodes = num_episodes
        self._no_previous_visit_policy = no_previous_visit_policy
        self._previous_visit_policy = previous_visit_policy
        self._model_trainer = model_trainer

        self._state_space = state_space.StateSpace(self._number_of_features)
        self._visited_states = set()
        self._state_value = np.zeros(2 ** self._number_of_features, dtype=np.float32)
        self._aor = self._initialize_AOR()

    def train(self) -> None:
        """Train the model for a given number of episodes."""
        for i in range(self._num_episodes):
            logger.info(f"Starting episode {i}")
            initial_state = self._state_space.get_initial_state()
            self.train_one_episode(initial_state)
            logger.info(f"self._aor: {self._aor}")

    def train_one_episode(self, initial_state: List[int]) -> None:
        """
        Train the model for one episode starting from the initial state.
        
        :param initial_state: The initial state to start the episode.
        """
        current_state = initial_state
        self._current_auprc = self._model_trainer.train_and_evaluate(self._X.loc[:, [bool(val) for val in current_state]], self._Y)
        logger.info(f"self._current_auprc: {self._current_auprc}")
        while True:
            # Check if the current state has been visited before
            logger.info(f"current_state: {current_state}")
            current_state_index = self._state_space.state_to_index(current_state)
            if current_state_index in self._visited_states:
                action = self._previous_visit_policy.select_action(current_state, self._aor)
            else:
                action = self._no_previous_visit_policy.select_action(current_state)

            # Take the selected action and get the new state and reward
            next_state_index, reward = self._take_action(current_state_index, action)
            logger.info(f"self._current_auprc: {self._current_auprc}")

            # Update the state value
            self._update_state_value(current_state_index, next_state_index, reward)

            self.update_AOR(action, next_state_index, current_state_index)

            # Update the current state
            current_state = self._state_space.index_to_state(next_state_index)

            # Check if the episode is a terminal state
            if all(current_state):
                break

    def _update_state_value(self, current_state_index: int, next_state_index: int, reward: float) -> None:
        """
        Update the state-value using Temporal Difference (TD) learning.
        
        :param current_state_index: The current state index.
        :param next_state_index: The new state index after taking an action.
        :param reward: The immediate reward received after transitioning from the current to the next state.
        """
        td_error = reward + self._state_value[next_state_index] - self._state_value[current_state_index]
        self._state_value[current_state_index] += self._alpha * td_error

    def _initialize_AOR(self) -> np.ndarray:
        """Initialize the Average of Rewards (AOR) for each feature."""
        # Initialize a 2 by n array, where n is the number of features.
        # The first row holds the counts, and the second row holds the AOR.
        return np.zeros((2, self._number_of_features), dtype=np.float32)

    def update_AOR(self, action: int, next_state_index: int, current_state_index: int) -> None:
        """Update the AOR value for a feature based on the state value."""
        # Retrieve the old count and old AOR value for the action (feature)
        old_count = self._aor[0, action]
        old_aor = self._aor[1, action]
        
        # Update the count
        new_count = old_count + 1
        self._aor[0, action] = new_count
        
        # Get the state values for the next state and the current state
        next_state_value = self._state_value[next_state_index]
        current_state_value = self._state_value[current_state_index]
        
        # Calculate the difference in state values
        state_value_diff = next_state_value - current_state_value
        
        # Update the AOR value
        new_aor = ((old_aor * old_count) + state_value_diff) / new_count
        self._aor[1, action] = new_aor

    def _take_action(self, current_state_index: int, action: int) -> Tuple[int, float]:
        """
        Take an action from the current state and return the next state index and the reward.
        
        :param current_state_index: The current state index.
        :param action: The action to be taken.
        :return: A tuple containing the next state index and the reward obtained.
        """
        next_state_index = self._state_transition(current_state_index, action)
        reward = self._calculate_reward(next_state_index)
        self._visited_states.add(next_state_index)
        return next_state_index, reward

    def _state_transition(self, current_state_index: int, action: int) -> int:
        """
        Transition from the current state to the next state based on the action taken.
        
        :param current_state_index: The current state index.
        :param action: The action taken (which is the index of the feature to be added).
        :return: The next state index.
        """
        current_state = self._state_space.index_to_state(current_state_index)
        current_state[action] = 1
        next_state_index = self._state_space.state_to_index(current_state)
        return next_state_index

    def _calculate_reward(self, next_state_index: int) -> float:
        """
        Calculate the reward for transitioning to next_state.

        :param next_state_index: The index of the next state.
        :return: The calculated reward.
        """
        # Train model with features from next_state and get accuracy
        next_state = self._state_space.index_to_state(next_state_index)
        next_auprc = self._model_trainer.train_and_evaluate(self._X.loc[:, [bool(val) for val in next_state]], self._Y)

        # Calculate reward as difference in accuracy
        reward = next_auprc - self._current_auprc

        # Update the current accuracy for the next transition
        self._current_auprc = next_auprc

        return reward

