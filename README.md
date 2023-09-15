# Feature Selection Using Reinforcement Learning

## Introduction

This project is an implementation of the paper ["Feature Selection Using Reinfocement Learning"](https://arxiv.org/pdf/2101.09460.pdf) and aims to tackle credit card fraud detection using Reinforcement Learning (RL). The RL agent learns optimal feature selection strategies to classify fraudulent transactions effectively.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing.

### Prerequisites

1. Clone the repository to your local machine.
    ```bash
    git clone <repository_url>
    ```
2. Navigate to the project directory.
    ```bash
    cd <project_directory>
    ```
3. Install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

### Usage

To execute feature selection, use the `FeatureSelector` class, which is inside the `feature_selector.py` module.

```python
from rl_feature_selector import feature_selector, policies, train


feature_selector = feature_selector.FeatureSelector(
    X=X,
    Y=Y,
    num_episodes=2_000,
    no_previous_visit_policy=policies.RandomPolicy(),
    previous_visit_policy=policies.EpsilonGreedyPolicy(epsilon=0.6),
    model_trainer=train.LinearSVMTrainer()
)
```

If you want to use a different classifier, implement the `ModelTrainer` trainer interface inside the `train.py` module. The same applies to the policies; modifications should implement the `Policy` interface inside the `policies.py` module.

The FeaatureSelector class expects the dataset to be splitted into features `X` and target `Y` as pandas DataFrames. The `dataset.py` module contains a utility class used to load the , but it should be replaced to use different datasets.

### Acknowledgments

- Thanks to the authors of the Kaggle credit card fraud detection dataset.
- Special thanks to all the open-source packages used in this project.
- Inspired by the research article "Feature selection using reinforcement learning" by Rasoul, Sali; Adewole, Sodiq; Akakpo, Alphonse (arXiv preprint arXiv:2101.09460, 2021).
