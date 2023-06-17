import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

def create_cnn_model(input_shape, num_classes):
    """
    Create a Convolutional Neural Network (CNN) model using TensorFlow.

    Parameters:
        input_shape (tuple): The shape of the input images in the format (height, width, channels).
        num_classes (int): The number of classes for classification.

    Returns:
        tensorflow.keras.Model: The compiled model.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# from class_names import get_class_names

# _, num_classes = get_class_names(train_dir='data/train')

# input_shape = (224, 224, 3)
# my_model = create_cnn_model(input_shape, num_classes)
# my_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# my_model.summary() 