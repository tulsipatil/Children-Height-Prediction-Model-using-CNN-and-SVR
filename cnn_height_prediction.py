import pandas as pd
import numpy as np
import cv2
import os
from ast import literal_eval

# Load Excel file containing height, depthmap image ID, and pose
data = pd.read_excel('height_and_pose.xlsx')

# Set image size and grayscale
img_size = (128, 128)
gray = True

# Create empty arrays for features and labels
X_img = []
X_pose = []
y = []

# Loop through each row in the CSV file
for i, row in data.iterrows():
    # Get height label, image ID, and pose
    height = row['Height(cm)']
    img_id = row['Depthmap Image']
    pose = row['Pose']
    
    # Read and preprocess image
    img_path = os.path.join('depthmap', img_id + '.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.0
    
    # Flatten the image and add it to the image features array
    X_img.append(img.flatten())
    
    # Convert pose string to dictionary and extract keypoint coordinates
    pose_dict = literal_eval(pose)[0]
    keypoints = pose_dict['key_points_coordinate']
    
    # Create array of pose coordinates
    pose_coords = []
    for kp in keypoints:
        kp_coord = [[kp[key]['x'], kp[key]['y']] for key in sorted(kp.keys())]
        pose_coords.extend(kp_coord)

        
    # Add pose coordinates to the pose features array
    X_pose.append(pose_coords)
    
    # Add the label to the labels array
    y.append(height)
    
# Convert lists to numpy arrays
X_img = np.array(X_img).reshape(-1, img_size[0], img_size[1], 1)
X_pose = np.array(X_pose)
y = np.array(y)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_img_train, X_img_test, X_pose_train, X_pose_test, y_train, y_test = train_test_split(X_img, X_pose, y, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

# Define the CNN architecture for image feature extraction
input_img = Input(shape=img_size+(1,))
x = Conv2D(32, (3,3), activation='relu')(input_img)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)
image_features = Dense(128, activation='relu')(x)

# Define the model architecture that combines image and pose features
input_pose = Input(shape=(34,))
x = concatenate([image_features, input_pose])
x = Dense(64, activation='relu')(x)
output = Dense(1)(x)
model = Model(inputs=[input_img, input_pose], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')
X_pose_train = X_pose_train.reshape(X_pose_train.shape[0], -1)

# Train the model
model.fit([X_img_train, X_pose_train], y_train, epochs=10, batch_size=32)

X_pose_train_flat = X_pose_train.reshape(X_pose_train.shape[0], -1)
X_pose_test_flat = X_pose_test.reshape(X_pose_test.shape[0], -1)

# Use the model to predict heights
y_pred = model.predict([X_img_test, X_pose_test_flat])

# Print predicted heights
print(y_pred)

from sklearn.metrics import mean_absolute_error

# Calculate MAE on test set
mae = mean_absolute_error(y_test, y_pred)

# Print MAE
print('Test MAE:', mae)

