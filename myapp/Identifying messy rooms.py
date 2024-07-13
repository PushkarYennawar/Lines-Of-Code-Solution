# Import libraries with explanations
import tensorflow as tf  # Deep learning framework

from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data augmentation
import numpy as np  # Numerical computations
from tensorflow.keras.preprocessing import image  # Image loading and processing

# Define data directories (check if they exist)
try:
  train_data_dir = "C:\\Users\\Parijat\\Downloads\\archive (5)\\images\\images_LOC\\train"
  test_data_dir = "C:\\Users\\Parijat\\Downloads\\archive (5)\\images\\images_LOC\\val"
except FileNotFoundError:
  print("Error: Data directories not found. Please check paths.")
  exit()

# Define class labels
class_labels = ["messy", "clean"]  # Modify these as needed

# Create data generators with augmentation techniques
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Flow data from directories with batch size and target size
training_set = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary",
    classes=class_labels
)

test_set = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode="binary",
    classes=class_labels
)

# Define the CNN model
cnn = tf.keras.models.Sequential()

# Convolutional layer 1 with ReLU activation (extracts features)
cnn.add(
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation="relu", input_shape=(64, 64, 3)
    )
)

# Max pooling layer (reduces dimensionality)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Convolutional layer 2 with ReLU activation (extracts more features)
cnn.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu")
)

# Max pooling layer (reduces dimensionality)
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flatten the output for dense layers
cnn.add(tf.keras.layers.Flatten())

# Dense layer 1 with ReLU activation (classification)
cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

# Output layer with sigmoid activation for binary classification
cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Model summary
cnn.summary()  # Print the model architecture

# Compile the model with optimizer, loss function, and metrics
cnn.compile(
    optimizer="adam",  # Optimization algorithm
    loss="binary_crossentropy",  # Loss function for binary classification
    metrics=["accuracy"],  # Monitor accuracy during training
)

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

# Train the model
cnn.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25,
    callbacks=[early_stopping]
)
