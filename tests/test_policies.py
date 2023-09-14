from unittest.mock import patch

import numpy as np

from rl_feature_selector import policies


def test_random_policy_select_action():
    policy = policies.RandomPolicy()
    state = [1, 1, 1, 1, 0]

    # Since the action is random, we can't predict the exact action.
    # Instead, we'll test if the action lies within the valid range.
    action = policy.select_action(state)
    assert action == 4
    assert state[action] == 0  # The selected feature should not already be in the state.

def test_epsilon_greedy_policy_select_action():
    epsilon = 0.2
    aor_mock = np.array([[0, 0, 0, 0, 0], [0.1, 0.2, 0.9, 0.8, 0.05]])
    policy = policies.EpsilonGreedyPolicy(epsilon=epsilon, aor=aor_mock)
    state = [0, 0, 1, 1, 0]
    
    # Mock the random number generator for exploration
    with patch('random.random', return_value=0.1):
        action = policy.select_action(state)
        assert state[action] == 0

    # Mock the random number generator for exploitation
    # Based on the given AOR values, the second feature should be selected,  because the third and fourth are already selected.
    with patch('random.random', return_value=0.9):
        action = policy.select_action(state)
        assert action == 1  

