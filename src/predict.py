import numpy as np
import pandas as pd

def predict(model, x_test):
    """Makes predictions on the test data."""
    predictions = model.predict(x_test)
    predicted_labels = np.argmax(predictions, axis=1)  # Get the digit with highest probability
    return predicted_labels

def create_submission_file(predicted_labels, submission_data, output_path="submission.csv"):
  """Creates a submission CSV file."""
  submission_data['Label'] = predicted_labels
  submission_data.to_csv(output_path, index=False)
  print(f"Submission file created at: {output_path}")