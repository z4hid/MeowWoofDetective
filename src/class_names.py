import numpy as np
import os
import pathlib

def get_class_names(train_dir):
    data_dir = pathlib.Path(train_dir)
    class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
    num_classes = len(class_names)
    return class_names, num_classes

# if __name__ == '__main__':
#     train_dir = '/home/zahid/Desktop/GitHub/MeowWoofDetective/data/train'
#     class_names, num_classes = get_class_names(train_dir)
#     print(class_names)
#     print(f'Number of Classes: {num_classes}')
