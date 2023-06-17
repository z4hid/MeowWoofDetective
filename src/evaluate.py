from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, classification_report
import numpy as np

def load_and_evaluate_model(model_path, test_data, class_names):
    """
    Load a trained model from a file and evaluate its performance on the test data.
    
    Args:
        model_path (str): Path to the saved model file.
        test_data (numpy.ndarray): Test data to be used for evaluation.
        class_names (list): List of class names.
        
    Returns:
        None
    """
    # Load the saved model
    model = load_model(model_path)
    print("Model Loaded")
    print("===================")
    
    # Perform predictions on the test data
    pred = model.predict(test_data)
    predicted_classes = np.argmax(pred, axis=1)
    true_classes = test_data.classes
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')
    f1score = f1_score(true_classes, predicted_classes, average='weighted')
    
    # Print evaluation metrics
    print(f'Model Accuracy: {accuracy}\n')
    print(f'Model Precision: {precision}\n')
    print(f'Model F1-Score: {f1score}\n')
    
    # Print classification report
    print(classification_report(true_classes, predicted_classes, target_names=class_names))




# # Example usage
# model_path = "models/cnn.h5"
# test_data = test_ds  # Assuming test_ds contains the test data


# load_and_evaluate_model(model_path, test_data, class_names)
