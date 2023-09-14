import random
from typing import List


class StateSpace:
    def __init__(self, number_of_features: int):
        self._number_of_features = number_of_features
    
    def state_to_index(self, state: List[int]) -> int:
        """
        Convert a state represented as a list of features to its corresponding index in the state-value array.
    
        :param state: State represented as a list of features.
        :return: Index in the state-value array.
        """
        state_str = ''.join(map(str, state))
        return int(state_str, 2)
    
    def index_to_state(self, index: int) -> List[int]:
        """
        Convert an index in the state-value array to its corresponding state represented as a list of features.
    
        :param index: Index in the state-value array.
        :return: State represented as a list of features.
        """
        state_str = format(index, '0' + str(self._number_of_features) + 'b')
        return [int(bit) for bit in state_str]
    
    def get_initial_state(self) -> List[int]:
        """
        Get an initial state for training an episode.
        
        :return: A list representing the initial state.
        """
        return [random.choice([0, 1]) for _ in range(self._number_of_features)]
