import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

class ConvLSTMModel:
    """
    Convolutional LSTM Model class for text classification.
    """

    def __init__(self, input_shape=(300, 171), num_classes=7):
        """
        Initializes the ConvLSTM model.

        Args:
            input_shape (tuple): Input shape of the model.
            num_classes (int): Number of classes for classification.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the ConvLSTM model architecture.
        """
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
        model.add(LSTM(64, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def compile_model(self, optimizer='adam', learning_rate=1e-5, weight_decay=5e-2):
        """
        Compiles the model with specified optimizer and learning rate.

        Args:
            optimizer (str): Name of the optimizer.
            learning_rate (float): Learning rate for optimization.
            weight_decay (float): Weight decay parameter.
        """
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=8, verbose=1):
        """
        Trains the model on the training data.

        Args:
            X_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            X_val (numpy.ndarray): Validation data features.
            y_val (numpy.ndarray): Validation data labels.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity mode.
        
        Returns:
            History: Training history.
        """
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
        return history

    def plot_training_history(self, history):
        """
        Plots the training history (loss and accuracy).

        Args:
            history (History): Training history.
        """
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')

        plt.show()

    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the model on the test data.

        Args:
            X_test (numpy.ndarray): Test data features.
            y_test (numpy.ndarray): Test data labels.
        
        Returns:
            tuple: Test loss and accuracy.
        """
        return self.model.evaluate(X_test, y_test)
