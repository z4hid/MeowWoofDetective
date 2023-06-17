import os
import matplotlib.pyplot as plt

def counter(root_dir):
    """
    Count the number of files in each subdirectory of a given directory and visualize the results.

    Parameters:
        root_dir (str): The directory path to be analyzed.

    Returns:
        None
    """
    number_label = {}  # A dictionary to store the count of files in each subdirectory
    total_files = 0  # Total count of files in all subdirectories

    for i in os.listdir(root_dir):  # Iterate over each subdirectory in the root directory
        counting = len(os.listdir(os.path.join(root_dir, i)))  # Count the number of files in the subdirectory
        number_label[i] = counting  # Store the count in the dictionary
        total_files += counting  # Update the total count

    print(f"Total Files: {str(total_files)}")  # Print the total count of files

    # Visualization using bar plot
    plt.figure(figsize=(5, 3))  # Create a figure with the specified size
    plt.bar(number_label.keys(), number_label.values())  # Plot the bar chart with subdirectory names and file counts
    plt.xticks(rotation='vertical')  # Rotate the x-axis labels vertically for better readability
    plt.title("Number of Images of Each Label")  # Set the title of the plot
    plt.xlabel('Label')  # Set the label for the x-axis
    plt.ylabel('Number of Images')  # Set the label for the y-axis
    plt.show()  # Display the plot

if __name__ == '__main__':
    train_dir = 'data/train'  # The directory path to be analyzed
    counter(train_dir)  # Call the counter function with the train directory as input
