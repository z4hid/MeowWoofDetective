import matplotlib.pyplot as plt
import random
import class_names

def visualization(X, Y=None):
    """
    Visualize a grid of images with optional labels.

    Parameters:
        X (numpy.ndarray): The array of images to be visualized.
        Y (numpy.ndarray, optional): The array of corresponding labels. Defaults to None.

    Returns:
        None
    """
    f = plt.figure(figsize=(20, 30))
    cols = 10
    rows = 10
    for i in range(0, 30):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('off')
        if Y is not None:
            label = random.choice(class_names)
            sp.set_title(label, fontsize=16)
        plt.imshow(X[i])

