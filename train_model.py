import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data for CNN input (28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,  # Random rotations
    zoom_range=0.1,  # Random zoom
    width_shift_range=0.1,  # Horizontal shift
    height_shift_range=0.1  # Vertical shift
)

# Fit the data generator on the training data
datagen.fit(x_train)

# Build the CNN model
model = Sequential([
    # Convolutional Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # Convolutional Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten the output for dense layer
    Flatten(),
    
    # Fully connected layer
    Dense(128, activation='relu'),
    Dropout(0.2),  # Adding Dropout to prevent overfitting
    
    # Output layer
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the data generator
model.fit(datagen.flow(x_train, y_train, batch_size=32),
          epochs=20,
          validation_data=(x_test, y_test),
          callbacks=[early_stopping])

# Save model
model.save("digit_model.h5")
print("✅ Model trained and saved as digit_model.h5")
