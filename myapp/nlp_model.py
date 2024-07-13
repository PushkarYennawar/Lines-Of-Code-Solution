# nlp_model.py
import tensorflow as tf
import numpy as np

# Load your pre-trained CNN model
def load_model():
    # Load your pre-trained model
    model = tf.keras.models.load_model('path_to_your_model')
    return model

def preprocess_image(image):
    # Preprocess the image (resize, normalize, etc.)
    # Here, you need to preprocess the image according to the requirements of your CNN model
    # For example:
    processed_image = tf.image.resize(image, [224, 224])  # Resize the image to match model input shape
    processed_image /= 255.0  # Normalize pixel values to [0, 1]
    return processed_image

def classify_image(image):
    model = load_model()  # Load the pre-trained model
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Expand the dimensions to match model input shape
    processed_image = np.expand_dims(processed_image, axis=0)
    
    # Run prediction on the preprocessed image using your model
    predictions = model.predict(processed_image)
    
    # Define class labels
    class_labels = ['cat', 'dog', 'other']
    
    # Get the index of the predicted class
    predicted_index = np.argmax(predictions, axis=1)[0]
    
    # Map the index to the corresponding class label
    predicted_class = class_labels[predicted_index]
    
    return predicted_class
