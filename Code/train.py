import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SpatialDropout2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# Function to load and preprocess the data
def load_data(data_path):
    images = []
    labels = []

    for filename in os.listdir(data_path):
        if filename.endswith(".png"):
            # Parse the label and image path from the filename
            label, _ = filename.split('_')
            label = int(label)
            
            # Read the image and resize if necessary
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            #img = cv2.resize(img, (image_width, image_height))

            # Normalize pixel values to between 0 and 1
            img = img / 255.0

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Set your image dimensions
image_width, image_height = 21, 31
output_layer_size = 64

# Set the path to your data
data_path = "../Data/characters/"

# Load and preprocess the data
images, labels = load_data(data_path)

# Encode Integer labels as inary arrays
encoded_labels = []
for label in labels:
    encode = []
    for i in range(6):
        encode.append(bool(label & 0b1))
        label = label >> 1
    encoded_labels.append(np.array(encode))
encoded_labels = np.array(encoded_labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# Reshape the data to match the input shape of the model
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


# Data Augmentation with ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0,
    height_shift_range=0,
    shear_range=0,
    zoom_range=0,
    preprocessing_function=lambda x: x + np.random.normal(0, 0.1, x.shape)  # Add random noise
)

epochs = 100
batch_size = 128

# Build the model
model = Sequential()
model.add(Conv2D(64, 5, activation='relu', input_shape=(image_height, image_width, 1)))
model.add(SpatialDropout2D(0.2))
model.add(Conv2D(32, 5, activation='relu', input_shape=(image_height, image_width, 1)))
model.add(SpatialDropout2D(0.2))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(6, activation='sigmoid'))  

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=batch_size), validation_data=(X_test, y_test), epochs=epochs)

# Save the trained model
model.save('braille_ocr_model.h5')

# make a prediction on the test set
yhat = model.predict(X_test)
# round probabilities to class labels
yhat = yhat.round()
# calculate accuracy
acc = accuracy_score(y_test, yhat)
# store result
print('>%.3f' % acc)