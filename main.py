import os
import argparse
import matplotlib.pyplot as plt

from data_loader import load_train_data, prepare_dataset, split_data
from model1 import BiLSTMModel
from model2 import ConvLSTMModel

class TrainModel:
    """
    Train BiLSTM or ConvLSTM model based on user input.

    Attributes:
        model1 (BiLSTMModel): Instance of BiLSTMModel for BiLSTM model.
        model2 (ConvLSTMModel): Instance of ConvLSTMModel for ConvLSTM model.
        X_train_data (numpy.ndarray): Input features for training data.
        y_train_data (numpy.ndarray): Labels for training data.
        X_val_data (numpy.ndarray): Input features for validation data.
        y_val_data (numpy.ndarray): Labels for validation data.
        X_test_data (numpy.ndarray): Input features for test data.
        y_test_data (numpy.ndarray): Labels for test data.
    """

    def __init__(self):
        """
        Initialize TrainModel with default values.
        """
        self.model1 = None
        self.model2 = None
        self.X_train_data = None
        self.y_train_data = None
        self.X_val_data = None
        self.y_val_data = None
        self.X_test_data = None
        self.y_test_data = None

    def train_model(self, train_json_file, model_type):
        """
        Train BiLSTM or ConvLSTM model.

        Args:
            train_json_file (str): Path to the train JSON file.
            model_type (str): 'bilstm' or 'convlstm'

        Returns:
            tuple: Paths to the saved loss and accuracy plots.
        """
        print("Loading training data from:", train_json_file)
        # Load training data
        train_data = load_train_data(train_json_file)

        print("Preparing dataset...")
        # Prepare dataset
        X_train, y_train = prepare_dataset(train_data)

        print("Splitting data into train, validation, and test sets...")
        # Split the data
        self.X_train_data, self.y_train_data, self.X_val_data, self.y_val_data, self.X_test_data, self.y_test_data = split_data(X_train, y_train)

        print(f"Initializing and compiling {model_type.upper()} model...")
        # Instantiate the model
        if model_type == 'bilstm':
            self.model1 = BiLSTMModel()
            self.model = self.model1
        elif model_type == 'convlstm':
            self.model2 = ConvLSTMModel()
            self.model = self.model2
        else:
            raise ValueError("Invalid model type. Choose 'bilstm' or 'convlstm'.")

        # Compile the model
        self.model.compile_model()

        print(f"Training {model_type.upper()} model...")
        # Train the model
        history = self.model.train_model(self.X_train_data, self.y_train_data, self.X_val_data, self.y_val_data)

        print("Plotting training history...")
        # Plot training history
        loss_plot_path, accuracy_plot_path = self.save_plots(model_type, history)

        # Test the trained model
        print(f"Evaluating the {model_type.upper()} model on the test set...")
        test_loss, test_accuracy = self.model.evaluate_model(self.X_test_data, self.y_test_data)
        print(f"{model_type.upper()} Test Loss:", test_loss)
        print(f"{model_type.upper()} Test Accuracy:", test_accuracy)

        print(f"Saving the {model_type.upper()} model...")
        # Save the model
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_type}_model.h5")
        self.model.model.save(model_path)

        return loss_plot_path, accuracy_plot_path

    def save_plots(self, model_type, history):
        """
        Save loss and accuracy plots.

        Args:
            model_type (str): 'bilstm' or 'convlstm'
            history (keras.callbacks.History): Training history object.

        Returns:
            tuple: Paths to the saved loss and accuracy plots.
        """
        # Save the loss plot
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        loss_plot_path = os.path.join(results_dir, f"{model_type}_loss_plot.png")
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{model_type.upper()} Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(loss_plot_path)
        plt.close()

        # Save the accuracy plot
        accuracy_plot_path = os.path.join(results_dir, f"{model_type}_accuracy_plot.png")
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{model_type.upper()} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(accuracy_plot_path)
        plt.close()

        return loss_plot_path, accuracy_plot_path

    def main(self):
        """Main function to parse command line arguments and start training."""
        parser = argparse.ArgumentParser(description='Train BiLSTM or ConvLSTM model')
        parser.add_argument('--train', type=str, choices=['bilstm', 'convlstm'], help='Train BiLSTM or ConvLSTM model')
        parser.add_argument('--train_file', type=str, help='Path to the train JSON file')
        args = parser.parse_args()

        if args.train:
            if args.train_file:
                loss_plot_path, accuracy_plot_path = self.train_model(args.train_file, args.train)
                print(f"Loss plot saved at: {loss_plot_path}")
                print(f"Accuracy plot saved at: {accuracy_plot_path}")
            else:
                print("Error: --train option requires --train_file argument.")
        else:
            print("Error: Please specify --train option.")

if __name__ == "__main__":
    TrainModel().main()
