import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import os

# Set environment variables (adjust memory limits for TensorFlow if needed)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# Path to dataset
path = './Resources'

# Data augmentation with ImageDataGenerator
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Added brightness augmentation
    validation_split=0.2  # Added validation split
)

# Load training and validation data
train_data = data_gen.flow_from_directory(
    path,
    target_size=(256, 256),
    batch_size=16,
    class_mode="categorical",
    subset='training'
)

val_data = data_gen.flow_from_directory(
    path,
    target_size=(256, 256),
    batch_size=16,
    class_mode="categorical",
    subset='validation'
)

# Define the model architecture
model = Sequential()

# Convolutional layers with batch normalization and pooling
model.add(Conv2D(256, (3, 3), activation="relu", input_shape=(256, 256, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2)))

# Flatten the output
model.add(Flatten())

# Fully connected layers with dropout
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(16, activation="relu"))

# Output layer (10 classes) with softmax activation
model.add(Dense(10, activation="softmax"))

# Compile the model with Adam optimizer and a custom learning rate
optimizer = Adam(learning_rate=0.0001)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Summary of the model
model.summary()

# Early stopping callback to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Increased patience

# Train the model with training and validation data
history = model.fit(
    train_data,
    validation_data=val_data,
    batch_size=16,
    epochs=50,  # Increased epochs for better convergence
    callbacks=[early_stop]
)

# Save the trained model
model.save('classifier_model_improved.keras')
