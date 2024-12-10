
# Mood Detection and Music Recommendation System

This project builds a web-based application that detects a user's mood using still images or videos. Based on the inferred mood, the system recommends personalized music to enhance the user's emotional experience. The application leverages Deep Learning (using TensorFlow/Keras) for mood detection and a Flask backend to provide the web interface and music recommendations.




## Project Features
1. Mood Detection:

- Detects user mood using a pre-trained Convolutional Neural Network (CNN) model from uploaded images or real-time video reactions.
- Supported moods include:
    - Happy
    - Sad
    - Energetic (angry, disgust, fear, surprise)
    - Calm (neutral)
2. Music Recommendation:

- Provides a list of music recommendations matching the detected mood.
- Reads the music data from a CSV file (data/data_moods.csv).
3. Web Interface:

- A simple Flask-based web interface for:
    - Uploading an image for mood detection.
    - Real-time video capturing to analyze facial expressions.
    - Displaying detected emotions, moods, and recommended songs.

## Installation and Setup
1. Prerequisites
Ensure you have the following software installed:

- Python (>=3.8)
- TensorFlow/Keras
- Flask
- OpenCV
- Pandas
- NumPy
2. Install Dependencies
Run the following command in your project directory to install all required libraries:
```
pip install tensorflow keras opencv-python flask pandas numpy
```
3. Folder Setup
    1. Place your pre-trained model file as ```My_model.h5``` in the project root directory.
    2. Prepare the ```data/data_moods.csv``` file with the following 
    3. Ensure your train and test folders contain grayscale face images categorized into the following classes:
- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

## Dataset
https://drive.google.com/drive/folders/1j9_rPzx0BGE4P8mhv6J0VvNPOhmIN8PU?usp=sharing
## Usage Instructions
1. Image Upload:

- Upload an image showing your face with any mood.
- Click submit to detect your emotion and get song recommendations.
2. Video Reaction:

- Allow camera access for real-time facial emotion detection.
- React with any mood for a specified duration (10 seconds).
- After the video ends, your detected emotion and recommended songs will be displayed.
## Model Architecture
The Convolutional Neural Network (CNN) architecture used for emotion detection includes:

- 4 convolutional blocks with ReLU activations and Batch Normalization.
- MaxPooling layers for downsampling.
- Dense layers for classification with Dropout for regularization.
- Softmax activation in the final layer to classify 7 emotions.
## Performance Metrics
The model was trained and evaluated on a dataset containing:

- Train Samples: 28,709 images
- Test Samples: 7,178 images
Results:

- Test Loss: 0.933
- Test Accuracy: 68%



## Dependencies
The following libraries are required:

- TensorFlow/Keras: Model training and inference
- OpenCV: Image and video processing
- Flask: Web framework
- Pandas: CSV handling
- NumPy: Numerical operations
- Matplotlib & Seaborn: Plotting (Confusion Matrix)

## Acknowledgments
- Dataset: FER2013 Emotion Dataset
- Tools: TensorFlow/Keras, OpenCV, Flask
