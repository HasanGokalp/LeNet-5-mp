import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
from keras.preprocessing.image import img_to_array, load_img

# Modelimiz
class LeNet(models.Sequential):
    """The LeNet-5 model."""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.add(layers.Conv2D(filters=6, kernel_size=5, activation='relu', padding='same', input_shape=(28, 28, 1)))
        self.add(layers.AvgPool2D(pool_size=2, strides=2))
        self.add(layers.Conv2D(filters=16, kernel_size=5, activation='relu'))
        self.add(layers.AvgPool2D(pool_size=2, strides=2))
        self.add(layers.Flatten())
        self.add(layers.Dense(120, activation='relu'))
        self.add(layers.Dense(84, activation='relu'))
        self.add(layers.Dense(num_classes))
        
#Fotoğrafı model için hazırlama
def preprocess_image(image_path, target_size=(28, 28)):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image /= 255.0  # Normalization
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Model tahminleri
def predict_with_lenet(model, image_path):
    image = preprocess_image(image_path)
    logits = model.predict(image)  # Model returns logits
    probabilities = tf.nn.softmax(logits).numpy()  # Apply softmax to logits
    predicted_class = np.argmax(probabilities, axis=1)
    return predicted_class, probabilities

# Load and preprocess MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = np.expand_dims(train_images / 255.0, -1)  # Normalize and expand dimension
test_images = np.expand_dims(test_images / 255.0, -1)    # Normalize and expand dimension

# Initialize and compile the model
model = LeNet(num_classes=10)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

# Assuming you have an image path
image_path = 'C:\\Users\\Hasan GÖKALP\\Desktop\\LenetImg\\deneme5.png'  # Update this path

# Predict with the model
predicted_class, probabilities = predict_with_lenet(model, image_path)
print(f"Predicted class: {predicted_class}")
print(f"Probabilities: {probabilities}")
