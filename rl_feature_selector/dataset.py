from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_credit_card_fraud_detection_dataset(
    dataset_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the Kaggle credit card fraud detection dataset from the given CSV path.

    This function reads the dataset from the provided path, applies a logarithmic transformation to the "Amount" column, scales the relevant columns using StandardScaler, and then splits the data into inputs (X) and targets (Y).

    :param dataset_path: Path to the CSV file containing the dataset.
    :return: A tuple containing two numpy arrays:
             - X: The input features with "Class" and "Amount" columns dropped.
             - Y: The target variable, which is the "Class" column.
    """
    df = pd.read_csv(dataset_path)
    df["Log_Amount"] = np.log1p(df["Amount"])

    # Separate positive and negative samples
    positive_samples = df[df["Class"] == 1]
    negative_samples = df[df["Class"] == 0]

    # Randomly sample from the negative samples
    sampled_negatives = negative_samples.sample(n=10_000)

    # Combine positive samples and sampled negatives. .sample(frac=1) shuffles the dataset.
    df = pd.concat([positive_samples, sampled_negatives]).sample(frac=1).reset_index(drop=True)

    # Data Normalization
    columns_to_scale = list(df.columns[1:29]) + ["Time", "Log_Amount"]
    df[columns_to_scale] = StandardScaler().fit_transform(df[columns_to_scale])

    # Split inputs and targets
    X = df.drop(columns=["Class", "Amount"])
    Y = df["Class"]
    return X, Y
