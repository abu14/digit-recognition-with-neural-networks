

## Handwritten Digit Recognition with Convolutional Neural Networks

![GitHub contributors](https://img.shields.io/github/contributors/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub forks](https://img.shields.io/github/forks/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub stars](https://img.shields.io/github/stars/abu14/Twitter-Sentiment-Analysis-Prediction?style=social)
![GitHub issues](https://img.shields.io/github/issues/abu14/Twitter-Sentiment-Analysis-Prediction)
![GitHub license](https://img.shields.io/github/license/abu14/Twitter-Sentiment-Analysis-Prediction)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abenezer-tesfaye-191579214/)

<p align="center">
  <img src="assets/digit_recognition.png" alt="Digit Recognition">
  
</p>

This is recognizing handwritten digits from the MNIST dataset using CNNs with 99.3 Accuracy Score. For an evern closer look on the steps taken please refer to the notebook.

## Project Structure

```
mnist-digit-recognition
├── data_loader.py       
├── data_preprocess.py  
├── model.py           
├── train.py            
├── predict.py      
└── main.py           
```


<!-- Tools Uses -->


## Tools Used

<p>
<img src="https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white">  
<img src="https://img.shields.io/badge/-Keras-D00000?style=flat&logo=keras&logoColor=white"> 
<img src="https://img.shields.io/badge/-scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/-NumPy-013243?style=flat&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/-Pandas-150458?style=flat&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/-Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white">
<img src="https://img.shields.io/badge/-Seaborn-3888E3?style=flat&logo=seaborn&logoColor=white">
</p>



## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/abu14/Digit-Recognition-CNN-99.3-Score.git  
    cd Digit-Recognition-CNN-99.3-Score  

    
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt  # Create a requirements.txt file with necessary libraries.
    ```
    `requirements.txt` 
    ```
    numpy
    pandas
    tensorflow
    scikit-learn
    ```

3.  **Download the Data:**

    The MNIST dataset is available on Kaggle (Digit Recognizer competition). You can download the `train.csv`, `test.csv`, and `sample_submission.csv` files. But I've placed them in the directory named `data` inside the project's root folder for your convenience.

4.  **Run the Main Script:**

    ```bash
    python main.py
    ```

    This will:

    *   Load the data.
    *   Preprocess the data.
    *   Create and train the CNN model.
    *   Evaluate the model's performance.
    *   Make predictions on the test data.
    *   Create a `submission.csv` file containing the predictions.

## Usage

The `main.py` script orchestrates the entire process.  Here's a breakdown of how they can be used individually:

1.  **Data Loading:**

    ```python
    from data_loader import load_data

    train_data, test_data, submission_data = load_data("data/train.csv", "data/test.csv", "data/sample_submission.csv")
    ```

2.  **Data Preprocessing:**

    ```python
    from data_preprocess import preprocess_data

    x_train, x_val, y_train, y_val, x_test = preprocess_data(train_data, test_data) #If you want to preprocess the test data too
    x_train, x_val, y_train, y_val = preprocess_data(train_data) #If you only want to preprocess the train data
    ```

3.  **Model Creation:**

    ```python
    from model import create_model

    model = create_model()
    ```

4.  **Model Training:**

    ```python
    from train import train_model

    history = train_model(model, x_train, y_train, x_val, y_val, epochs=40, batch_size=32)  # Adjust epochs and batch size as needed
    ```

5.  **Prediction:**

    ```python
    from predict import predict, create_submission_file

    predicted_labels = predict(model, x_test)
    create_submission_file(predicted_labels, submission_data)

    ```

## Model Architecture

The CNN model architecture consists of:

*   Convolutional layers with ReLU activation.
*   Batch normalization layers.
*   Max pooling layers.
*   Dropout layers for regularization.
*   Fully connected (dense) layers.
*   A softmax output layer for classification.

## Training

*   **Optimizer:** Adam optimizer.
*   **Loss Function:** Categorical cross-entropy.
*   **Data Augmentation:**  `ImageDataGenerator` is used for real-time data augmentation during training.
*   **Learning Rate Scheduling:** `LearningRateScheduler` is used to adjust the learning rate during training.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues. I respond on all of my socials so feel free to reach out.

## License
<!-- LICENSE -->
This project is licensed under the MIT License. See [LICENSE](./LICENSE) file for more details.

<!-- CONTACT -->
## **Contact**

##### Abenezer Tesfaye

⭐️ Email - tesfayeabenezer64@gmail.com
 
Project Link: [Github Repo](https://github.com/abu14/Digit-Recognition-CNN-99.3-Score)
