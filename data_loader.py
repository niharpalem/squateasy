
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_train_data(train_json_file):
    """
    Load training data from a JSON file.

    Parameters:
    - train_json_file (str): Path to the JSON file containing training data.

    Returns:
    - train_data (dict): Dictionary containing the loaded training data.
    """
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    return train_data

def extract_3d_joint(json_file_path):
    """
    Extract 3D joint coordinates from a JSON file.

    Parameters:
    - json_file_path (str): Path to the JSON file containing 3D joint coordinates.

    Returns:
    - joint_coordinates (list): List of 3D joint coordinates.
    """
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    joint_coordinates = []
    for frame_index, frame_data in data.items():
        if '3d_joint' in frame_data:
            joint_coordinates.append(frame_data['3d_joint'])

    return joint_coordinates

def prepare_dataset(train_data):
    """
    Prepare dataset from the loaded training data.

    Parameters:
    - train_data (dict): Dictionary containing the loaded training data.

    Returns:
    - X_train (list): List of input features (3D joint coordinates).
    - y_train (numpy.ndarray): Array of labels.
    """
    X_train = []
    y_train = []
    for index, info in train_data.items():
        folder_path = info["folder_path"]
        label = int(info["label"])

        # Construct the full path to the JSON file
        json_file_path = os.path.join("DATA", os.path.basename(folder_path))

        # Extract the 3D joint coordinates
        joint_coordinates = extract_3d_joint(json_file_path)

        # Append the 3D joint coordinates to the training data
        X_train.append(joint_coordinates)
        y_train.append(label)

    # Convert y_train to a numpy array
    y_train = np.array(y_train)

    return X_train, y_train

def split_data(X, y, train_ratio=0.6, validation_ratio=0.2, test_ratio=0.2):
    """
    Split data into training, validation, and test sets.

    Parameters:
    - X (list): List of input features.
    - y (numpy.ndarray): Array of labels.
    - train_ratio (float): Ratio of training data.
    - validation_ratio (float): Ratio of validation data.
    - test_ratio (float): Ratio of test data.

    Returns:
    - X_train_data (numpy.ndarray): Array of training input features.
    - y_train_data (numpy.ndarray): Array of training labels.
    - X_val_data (numpy.ndarray): Array of validation input features.
    - y_val_data (numpy.ndarray): Array of validation labels.
    - X_test_data (numpy.ndarray): Array of test input features.
    - y_test_data (numpy.ndarray): Array of test labels.
    """
    # First, split the data into training and remaining data with their labels
    X_train_data, X_temp_data, y_train_data, y_temp_data = train_test_split(
        X, y, test_size=1 - train_ratio, stratify=y, random_state=42)

    # Now split the remaining data into validation and test sets
    X_val_data, X_test_data, y_val_data, y_test_data = train_test_split(
        X_temp_data, y_temp_data, test_size=test_ratio/(test_ratio + validation_ratio), stratify=y_temp_data, random_state=42)

    # If you need to truncate or process sequences, adjust here
    # Truncate sequences to a maximum of 300 elements for each example
    X_train_data = [seq[:300] for seq in X_train_data]
    X_val_data = [seq[:300] for seq in X_val_data]
    X_test_data = [seq[:300] for seq in X_test_data]

    # Convert all to numpy arrays for compatibility with Keras/TensorFlow
    X_train_data = np.array(X_train_data)
    X_val_data = np.array(X_val_data)
    X_test_data = np.array(X_test_data)

    return X_train_data, y_train_data, X_val_data, y_val_data, X_test_data, y_test_data
