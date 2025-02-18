import pandas as pd
import numpy as np

def load_data(train_path, test_path, submission_path):
    """Loads and prepares the train, test, and submission data.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the test CSV file.
        submission_path (str): Path to the submission CSV file.

    Returns:
        tuple: A tuple containing the training data (NumPy array),
               the test data (Pandas DataFrame), and the submission
               data (Pandas DataFrame).
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    submission_data = pd.read_csv(submission_path)

    # Load training data as NumPy array (more efficient for numerical work)
    train_np = np.loadtxt(train_path, skiprows=1, dtype='int', delimiter=',')
    return train_np, test_data, submission_data