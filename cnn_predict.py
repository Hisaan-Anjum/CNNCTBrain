import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras as keras
def load_images(directory):
    images = []
    for file in os.listdir(directory):
        if file.endswith(('.jpg', '.png')):
            image_file = os.path.join(directory, file)
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            images.append(img)
    return np.array(images)

def preprocess_data(images):
    images = images.reshape(-1, 256, 256, 1)  # Add channel dimension for grayscale images
    images = images.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
    return images

def predict_with_cnn(model_filename, test_images_directory):
    loaded_model = load_model(model_filename)
    test_images = load_images(test_images_directory)
    test_images = preprocess_data(test_images)

    probabilities = loaded_model.predict(test_images)

    normal_count = 0
    abnormal_count = 0
    for i, prob in enumerate(probabilities):
        if prob <= 0.5:
            class_label = 'Normal'
            normal_count += 1
        else:
            class_label = 'Abnormal'
            abnormal_count += 1
        print(f"Image {i + 1} - Predicted Class: {class_label}")
        print(f"Image {i + 1} - Probability: {prob}")
    print(f"Normal: {normal_count}")
    print(f"Abnormal: {abnormal_count}")

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    model_filename = 'ct_anomaly_detection_cnn.h5'
    test_images_directory = 'code_testing_abnormal'
    predict_with_cnn(model_filename, test_images_directory)
