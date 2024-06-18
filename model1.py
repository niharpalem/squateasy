import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np

class BiLSTMModel:
    """BiLSTM Model class for sentiment analysis."""

    def __init__(self, input_shape=(300, 171), num_classes=7, l2_reg=0.001):
        """
        Initialize the BiLSTM model.

        Args:
            input_shape (tuple): Shape of the input data. Default is (300, 171).
            num_classes (int): Number of output classes. Default is 7.
            l2_reg (float): L2 regularization factor. Default is 0.001.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.model = self._build_model()

    def _build_model(self):
        """Build the BiLSTM model architecture."""
        model = Sequential()
        model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(self.l2_reg)), input_shape=self.input_shape))
        model.add(Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(self.l2_reg))))
        model.add(Dropout(0.25))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        return model

    def compile_model(self, optimizer='adam', learning_rate=1e-4, weight_decay=9E-2):
        """
        Compile the BiLSTM model.

        Args:
            optimizer (str): Name of the optimizer. Default is 'adam'.
            learning_rate (float): Learning rate for the optimizer. Default is 1e-4.
            weight_decay (float): Weight decay parameter. Default is 9E-2.
        """
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, weight_decay=weight_decay)
        self.model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=10, verbose=1):
        """
        Train the BiLSTM model.

        Args:
            X_train (numpy.ndarray): Training data.
            y_train (numpy.ndarray): Training labels.
            X_val (numpy.ndarray): Validation data.
            y_val (numpy.ndarray): Validation labels.
            epochs (int): Number of epochs for training. Default is 50.
            batch_size (int): Batch size for training. Default is 10.
            verbose (int): Verbosity mode (0, 1, or 2). Default is 1.

        Returns:
            object: History object containing training history.
        """
        history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
        return history

    def plot_training_history(self, history):
        """
        Plot the training history.

        Args:
            history (object): History object returned by the training process.
        """
        plt.figure(figsize=(10, 8))
        plt.plot(history.history['loss'], label='Training loss', linestyle='-')
        plt.plot(history.history['val_loss'], label='Validation loss', linestyle='-')

        window_size = 5
        loss_train_smooth = np.convolve(history.history['loss'], np.ones(window_size) / window_size, mode='valid')
        loss_val_smooth = np.convolve(history.history['val_loss'], np.ones(window_size) / window_size, mode='valid')

        plt.plot(np.arange(window_size - 1, len(history.history['loss'])), loss_train_smooth, label='Training loss (smoothed)', linestyle='--')
        plt.plot(np.arange(window_size - 1, len(history.history['val_loss'])), loss_val_smooth, label='Validation loss (smoothed)', linestyle='--')

        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.plot(history.history['accuracy'], label='Training accuracy', linestyle='-')
        plt.plot(history.history['val_accuracy'], label='Validation accuracy', linestyle='-')

        accuracy_train_smooth = np.convolve(history.history['accuracy'], np.ones(window_size) / window_size, mode='valid')
        accuracy_val_smooth = np.convolve(history.history['val_accuracy'], np.ones(window_size) / window_size, mode='valid')

        plt.plot(np.arange(window_size - 1, len(history.history['accuracy'])), accuracy_train_smooth, label='Training accuracy (smoothed)', linestyle='--')
        plt.plot(np.arange(window_size - 1, len(history.history['val_accuracy'])), accuracy_val_smooth, label='Validation accuracy (smoothed)', linestyle='--')

        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the BiLSTM model on the test set.

        Args:
            X_test (numpy.ndarray): Test data.
            y_test (numpy.ndarray): Test labels.

        Returns:
            list: Test loss and accuracy.
        """
        return self.model.evaluate(X_test, y_test)
