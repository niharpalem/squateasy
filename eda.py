import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_train_data, prepare_dataset, split_data

class EDA:
    """Class for Exploratory Data Analysis (EDA)"""

    def __init__(self, y_train_data, y_val_data, y_test_data):
        """
        Initialize EDA object.

        Parameters:
        - y_train_data (numpy.ndarray): Array of training labels.
        - y_val_data (numpy.ndarray): Array of validation labels.
        - y_test_data (numpy.ndarray): Array of test labels.
        """
        self.y_train_data = y_train_data
        self.y_val_data = y_val_data
        self.y_test_data = y_test_data
        self.create_eda_dir()

    def create_eda_dir(self):
        """Create a directory for saving EDA plots if it doesn't exist."""
        if not os.path.exists("eda"):
            os.makedirs("eda")
     
    def plot_class_distribution(self):
        """Plot the class distribution before and after splitting the data."""
        plt.figure(figsize=(15, 6))
    
        # Before splitting
        plt.subplot(1, 2, 1)
        # Calculate class counts before splitting
        class_counts = np.unique(np.concatenate((self.y_train_data, self.y_val_data, self.y_test_data)), return_counts=True)
        plt.bar(class_counts[0], class_counts[1], color='b')
        plt.title('Class Distribution (Before Splitting)')
        plt.xlabel('Class')
        plt.ylabel('Count')
    
        # After splitting
        plt.subplot(1, 2, 2)
        # Calculate class counts after splitting
        class_counts_train = np.unique(self.y_train_data, return_counts=True)
        class_counts_val = np.unique(self.y_val_data, return_counts=True)
        class_counts_test = np.unique(self.y_test_data, return_counts=True)
        classes = np.unique(np.concatenate((self.y_train_data, self.y_val_data, self.y_test_data)))
    
        width = 0.2
        x = np.arange(len(classes))
        plt.bar(x - width, class_counts_train[1], width, label='Train', color='b')
        plt.bar(x, class_counts_val[1], width, label='Validation', color='g')
        plt.bar(x + width, class_counts_test[1], width, label='Test', color='r')
        plt.xticks(x, classes)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Class Distribution (After Splitting)')
        plt.legend()
    
        plt.tight_layout()
        plt.savefig("eda/class_distribution_before_after_splitting.png")
        plt.show()
    
        

   


if __name__ == "__main__":
    train_json_file = 'Divided/train.json'
    train_data = load_train_data(train_json_file)
    X_train, y_train = prepare_dataset(train_data)
    X_train_data, y_train_data, X_val_data, y_val_data, X_test_data, y_test_data = split_data(X_train, y_train)

    eda = EDA(y_train_data, y_val_data, y_test_data)
    eda.plot_class_distribution()
