# cnn_model.py
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, val_dir):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    training_set = train_datagen.flow_from_directory(train_dir,
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(val_dir,
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')
    return training_set, test_set

def build_cnn():
    cnn = tf.keras.models.Sequential()
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    return cnn

def train_cnn(cnn, training_set, test_set):
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)

def predict_single_image(cnn, image_path):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict(test_image)
    if result[0][0] == 0:
        prediction = 'clean'
    else:
        prediction = 'unclean'
    return prediction

if __name__ == "__main__":
    train_dir = 'C:\Users\Ashwani\Documents\archive (7)[2]'
    val_dir = 'C:\Users\Ashwani\Documents\archive (7)[2]\val'
    
    training_set, test_set = load_data(train_dir, val_dir)
    cnn_model = build_cnn()
    train_cnn(cnn_model, training_set, test_set)
    
    test_image_path = "D:\\loc\\images\\images\\train\\clean\\11.png"
    prediction = predict_single_image(cnn_model, test_image_path)
    print("Prediction:", prediction)
