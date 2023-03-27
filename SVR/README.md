# Children Height Prediction Model

This code is an implementation of a height prediction model using a support vector regression (SVR) algorithm. The model takes depthmap images as inputs and predicts the corresponding heights in centimeters.


## Requirements

This code uses the following libraries:

* Pandas

* Numpy
* OpenCV (cv2)
* Scikit-learn

Please ensure that these libraries are installed before running the code.
## Installation

1. Install Python 3.x from python.org
2. Open a command prompt or terminal window and install the required packages using the following command:

```bash
  pip install pandas numpy opencv-python scikit-learn
```
    
## Usage

1. Clone or download the source code and navigate to the directory.
2. Ensure that the Excel file containing the height and depthmap image ID is in the same directory as the source code.
3. Run the program using the following command:

```javascript
python height_and_depthmap.py
```
4. The program will output the mean absolute error (MAE) of the model on the test set and the predicted heights.

## Data
The input data for this model is a Microsoft Excel file containing two columns:

* Depthmap Image: The ID of the depthmap image file (without the file extension)
* Height(cm): The height corresponding to the depthmap image ID
The depthmap image files should be located in a folder named depthmap in the same directory as the code file.

## Preprocessing
The depthmap images are read using the OpenCV library and preprocessed by resizing to 128x128 pixels and converting to grayscale if specified. The images are then flattened and normalized before being fed into the model.

## Model
The SVR model is defined with a linear kernel. The model is trained using the training data and evaluated using the mean absolute error (MAE) on the test set.

## Output
The model output consists of two parts:

* The mean absolute error (MAE) on the test set, which is printed to the console.
* The predicted heights corresponding to the test set, which are also printed to the console.
## Approach
The solution uses the Support Vector Regression (SVR) algorithm to train the model. The depthmap images are read and preprocessed using OpenCV library. The images are resized to (128, 128) pixels and converted to grayscale. The flattened images are used as features, and the corresponding heights are used as labels for training the model. The SVR algorithm with a linear kernel is used to train the model. The mean absolute error (MAE) is used as the evaluation metric for the model.

The reason for choosing this approach is that it can handle non-linear data and performs well with small datasets. The SVR algorithm is also less prone to overfitting and can work well with high-dimensional data.

## Conclusion

The provided solution offers a way to predict the height of children below the age of 5 years using depthmap images. The code is easy to run, and the approach used can be modified and used for other similar tasks.
## References

* OpenCV documentation: https://docs.opencv.org/master/
* scikit-learn documentation: https://scikit-learn.org/stable/documentation.html