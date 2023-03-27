'''Approach
The solution uses the Support Vector Regression (SVR) algorithm to train the model. The depthmap images are read and preprocessed using OpenCV library. The images are resized to (128, 128) pixels and converted to grayscale. The flattened images are used as features, and the corresponding heights are used as labels for training the model. The SVR algorithm with a linear kernel is used to train the model. The mean absolute error (MAE) is used as the evaluation metric for the model.

The reason for choosing this approach is that it can handle non-linear data and performs well with small datasets. The SVR algorithm is also less prone to overfitting and can work well with high-dimensional data.'''

import pandas as pd
import numpy as np
import cv2
import os

# Load Excel file containing height and depthmap image ID
data = pd.read_excel('height_and_pose.xlsx')

# Set image size and grayscale
img_size = (128, 128)
gray = True

# Create empty arrays for features and labels
X = []
y = []

# Loop through each row in the CSV file
for i, row in data.iterrows():
    # Get height label and image ID
    height = row['Height(cm)']
    img_id = row['Depthmap Image']
    
    # Read and preprocess image
    img_path = os.path.join('depthmap', img_id + '.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    
    # Flatten the image and add it to the features array
    X.append(img.flatten())
    # Add the label to the labels array
    y.append(height)
    
# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVR

# Define the model architecture
model = SVR(kernel='linear')

# Train the model
model.fit(X_train, y_train)

# Evaluate the model on the test set
test_predictions = model.predict(X_test)

from sklearn.metrics import mean_absolute_error
test_mae = mean_absolute_error(y_test, test_predictions)
print('Test MAE:', test_mae)

# Print the predicted heights
print('Predicted heights:', test_predictions)
