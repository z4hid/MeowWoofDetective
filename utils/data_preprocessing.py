import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Get the class names for our multi-class dataset
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)
num_classes = len(class_names)
print(f'Number of Classes: {num_classes}')


def preprocess_data(train_dir, test_dir, image_size, batch_size):
    # Data augmentation and normalization for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Normalization for test set (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load and preprocess training data
    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # Load and preprocess test data
    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    return train_data, test_data
