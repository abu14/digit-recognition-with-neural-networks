import os
from .data_loader import load_data
from .data_preprocess import preprocess_data
from .model import create_model
from .train import train_model
from .predict import predict, create_submission_file

## if you plan on using these paths, make sure to change them to the path on you local
TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUBMISSION_PATH = 'data/sample_submission.csv' #Use this file to predict and then save the ouput and submit to the competition on Kaggle
OUTPUT_PATH = "submission.csv"

def main():
    train_data, test_data, submission_data = load_data(TRAIN_PATH, TEST_PATH, SUBMISSION_PATH)
    x_train, x_val, y_train, y_val, x_test = preprocess_data(train_data, test_data) # Pass test_data

    model = create_model()
    history = train_model(model, x_train, y_train, x_val, y_val)

    final_loss, final_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"Model - Final Loss: {final_loss}, and Final Accuracy: {final_acc}")

    predicted_labels = predict(model, x_test)
    create_submission_file(predicted_labels, submission_data, OUTPUT_PATH)

if __name__ == "__main__":
    main()