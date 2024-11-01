import numpy as np

import os

import cv2

import tensorflow as tf

2

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,

Dense, Dropout

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# Define image size and batch size

img_size = 64

batch_size = 32

# Load and preprocess data

def load_data(data_dir):

images = []

labels = []

calorie_info = {} # Dictionary to hold calorie info

for label in os.listdir(data_dir):

class_dir = os.path.join(data_dir, label)

calorie_count = get_calories(label) # Implement this function to get calorie count

calorie_info[label] = calorie_count

for img_name in os.listdir(class_dir):

img_path = os.path.join(class_dir, img_name)
img = cv2.imread(img_path)

img = cv2.resize(img, (img_size, img_size))

images.append(img)

labels.append(label)

return np.array(images) / 255.0, np.array(labels), calorie_info

# Dummy function to get calorie count for each food item

def get_calories(label):

calorie_dict = {

'apple': 95,

'banana': 105,

'burger': 354,

# Add more food items and their calorie counts here

}

return calorie_dict.get(label, 0)

data_dir = 'path_to_your_data' # Replace with your dataset path

X, y, calorie_info = load_data(data_dir)

# Encode labels

class_names = np.unique(y)

y_encoded = np.array([np.where(c p.array([np.where(class_names == label)[0][0] for label in y])
# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model

model = Sequential([

Conv2D(32, (3, 3), activation 'relu', input_shape=(img_size, img_size, 3)),

MaxPooling2D(pool_size=(2, 2)),

Conv2D(64, (3, 3), activation 'relu'),

MaxPooling2D(pool_size=(2, 2)),

Flatten(),

Dense(128, activation='relu'),

Dropout(0.5),

Dense(len(class_names), activation='softmax') # Number of classes ])

# Compile the model

model.compile(optimizer='adam',

loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=batch_size)

# Evaluate the model

test_loss, test_acc = model.evaluate(X_test, y_test) print(f'Test accuracy: {test_acc}')

# Save the model model.save('food_recognition_model.h5')

# Function to predict food and estimate calories

def predict_food(image_path): img = cv2.imread(image_path) img = cv2.resize(img, (img_size, img_size)) / 255.0 img = np.expand_dims(img, axis=0)

predictions = model.predict(img)

predicted_class = np.argmax(predictions)

food_item = class_names[predicted_class]

calories = calorie_info[food_item]

return food_item, calories

# Example usage
food, calories = predict_food('path_to_your_image.jpg') # Replace with your image path

print(f'Food: (food), Calories: {calories}')

# Plot training history

plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label='val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
