import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def preprocess_data(train_data, test_data=None):  # Added test_data argument
    """Preprocesses the training data (and optionally the test data).

    Args:
        train_data (numpy.ndarray): The training data as a NumPy array.
        test_data (pandas.DataFrame, optional): The test data as a DataFrame.

    Returns:
        tuple: A tuple containing the preprocessed training data (x_train, y_train),
               and optionally the preprocessed test data (x_test).
    """

    X = train_data[:, 1:]  # Features (pixel values)
    y = train_data[:, 0]   # Labels (digits)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42) # Added random_state

    # Reshape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_val = x_val.reshape(-1, 28, 28, 1)

    # Standardize
    x_train = x_train.astype('float32') / 255.0  # Use 255.0 for float division
    x_val = x_val.astype('float32') / 255.0

    # One-hot encode
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    if test_data is not None:  # Preprocess test data if provided
        x_test = test_data.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return x_train, x_val, y_train, y_val, x_test
    else:
        return x_train, x_val, y_train, y_val