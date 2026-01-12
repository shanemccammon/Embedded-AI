# Embedded AI Lab 3 — Magic Wand Gesture Recognition [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ColoradoEmbeddedAI-Fall2025/lab-3-building-a-dataset-team6/blob/main/model/model.ipynb)

### Team 6

* **Alha Kane**
* **Matt Hartnett**
* **Piyush Nagpal**
* **Sam Walker**
* **Shane McCammon**
* **Zane McMorris**

---

## Project Overview

This project implements a **gesture recognition system** that transforms a smartphone into a **magic wand** using AI.
The trained model classifies two motion gestures based on accelerometer and orientation data collected from smartphones:

* **Square**
* **Triangle**

---

## Data Collection

Data was collected using the [**Sensor Logger**](https://www.tszheichoi.com/sensorlogger) app on smartphones. In order to achieve the best results, faster movements were used. I.E. drawing the entire gestures in 2-4s. This is because the snr is much better because the noise remains the same while the acceleration is much larger.

### Data Details

* **Sensors used:** Accelerometer, Orientation
* **Sampling rate:** ~100 Hz
* **Format:** CSV files with time-synchronized IMU readings
* **Preprocessing:**

  * Data normalization
  * Segmentation into fixed-size windows
  * Label encoding (`0` → Square, `1` → Triangle)

### Data Organization

* Data
  * Training
    * Square
      * .zip (orientation and accelerometer data)
    * Triangle
      * .zip (orientation and accelerometer data)
  * Validation
    * Square
      * .zip (orientation and accelerometer data)
    * Triangle
      * .zip (orientation and accelerometer data) 


We have a python script that parses all the zip files for us. This allows us to simply upload the corresponding zip files straight from the app to the appropriate folders.

---

## Model Architecture

The model is a convolutional neural network (CNN) designed for multi-class image classification. It processes input images of shape `(32, 32, 3)` and outputs predictions over `2 (square, triangle)` categories.

### Architecture Details:

* **Input Layer:** Accepts RGB images with specified width and height.
* **Rescaling:** Input pixel values are normalized by scaling them to the [0, 1] range.
* **Convolutional Blocks:**

  * Three convolutional layers with filter sizes 16, 32, and 64 respectively.
  * Each convolution uses a 3x3 kernel with stride 2 and "same" padding, effectively reducing spatial dimensions.
  * Each conv layer is followed by batch normalization and ReLU activation.
  * Dropout of 0.4 is applied after each activation to reduce overfitting.
* **Global Average Pooling:** Reduces the feature maps to a single vector per channel, aggregating spatial information.
* **Dropout Layer:** A final dropout with rate 0.5 to further regularize the model.
* **Output Layer:** Fully connected dense layer with `2` units and a softmax activation for multi-class probability output.

### Training Setup:

* **Loss Function:** Sparse categorical crossentropy, suitable for integer-labeled multi-class classification.
* **Optimizer:** Adam with a learning rate of 0.001.
* **Metrics:** Accuracy to evaluate performance.
* **Epochs:** The model is trained for 50 epochs.
* **Data:** Training and validation datasets are preprocessed and fed through TensorFlow datasets with prefetching for performance.

---

## Setup Instructions

### Requirements

* `python >= 3.12`
* `uv` (virtual environment manager)

### Environment Setup

```bash
# Create and sync virtual environment
uv sync
source ./.venv/bin/activate
```

Then open and run the notebook:

```bash
jupyter notebook model/model.ipynb
```

---

## Model Evaluation

### Confusion Matrix

| Actual \ Predicted | Square | Triangle |
| ------------------ | :----: | :------: |
| **Square**         |  0.98  |   0.02   |
| **Triangle**       |  0.04  |   0.96   |

### Results Summary

* **Accuracy:** ~97%
* **Precision (Square):** 0.98
* **Precision (Triangle):** 0.96
* **Loss:** < 0.1 after 50 epochs
* **Observation:** Excellent class separation with minimal misclassification.
