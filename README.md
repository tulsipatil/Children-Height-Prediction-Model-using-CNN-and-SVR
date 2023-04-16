# Children-Height-Prediction-Model-using-CNN-and-SVR

This repository contains two models that use different approaches to predict the height of children based on depthmap images and pose information. The models are implemented using Python and TensorFlow/Keras and Scikit-learn libraries.

## Dataset
The dataset used in this project consists of depthmap images and pose information for children of different ages. The dataset also includes the height of each child in centimeters.

## Models
### Convolutional Neural Network (CNN)
The first model is a convolutional neural network (CNN) that extracts image features from the depthmap images and combines them with pose information to predict the height of children. The architecture of the CNN includes two convolutional layers followed by two max-pooling layers and a fully connected layer.

### Support Vector Regression (SVR)
The second model is a support vector regression (SVR) model that takes depthmap images as inputs and predicts the corresponding heights in centimeters.

## Usage
To use the models, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries.
3. Load your own dataset or use the provided dataset by placing your depthmap images in the depthmap directory and updating the height_and_pose.xlsx file with the corresponding image IDs, pose information, and height labels.
4. Run the models by executing the cnn_height_prediction.py or svr_height_prediction.py files, depending on which model you want to use.
5. View the predictions and mean absolute error (MAE) in the console output.
