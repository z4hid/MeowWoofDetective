import numpy as np
import pathlib

def get_class_names(data_dir):
    """
    Get the class names for a multi-class dataset.

    Parameters:
        data_dir (str): The directory path to the dataset.

    Returns:
        class_names (numpy.ndarray): An array of class names.
        num_classes (int): The number of classes in the dataset.
    """
    data_dir = pathlib.Path(data_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    num_classes = len(class_names)
    return class_names, num_classes




