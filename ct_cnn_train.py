import os
import numpy as np
import cv2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def load_images(directory):
    images = []
    for file in os.listdir(directory):
        if file.endswith(('.jpg', '.png')):
            image_file = os.path.join(directory, file)
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            images.append(img)
    return np.array(images)

def preprocess_images(images):
    # Convert images to floating point and normalize
    images = images.astype('float32') / 255.0
    # Expand dimensions to make them compatible with CNN input shape
    images = np.expand_dims(images, axis=-1)
    return images

def build_cnn(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for anomaly detection
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    normal_images_directory = 'brain-ct-hemorrhage-dataset/Training/NORMAL'
    anomalous_images_directory = 'brain-ct-hemorrhage-dataset/Training/Hemorrhagic'

    normal_images = load_images(normal_images_directory)
    anomalous_images = load_images(anomalous_images_directory)

    images = np.concatenate((normal_images, anomalous_images), axis=0)
    labels = np.concatenate((np.zeros(len(normal_images)), np.ones(len(anomalous_images))), axis=0)

    print("Number of normal images:", len(normal_images))
    print("Number of anomalous images:", len(anomalous_images))

    # Preprocess images
    images = preprocess_images(images)

    input_shape = images[0].shape
    cnn_model = build_cnn(input_shape)

    # Train the CNN on all available data
    cnn_model.fit(images, labels, epochs=10, batch_size=32)

    # Save the trained model
    cnn_model.save('anomaly_detection_cnn.h5')

    print('Trained & saved!')
