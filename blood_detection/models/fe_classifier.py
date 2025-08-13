"""
Fast Extraction (FE) Classification Model
==========================================

Implementation of the classification model from the Fast Extraction framework
for blood detection in hyperspectral images.

Based on the paper: "Enhancing forensic blood detection using hyperspectral imaging 
and advanced preprocessing techniques"

The model consists of:
- Two Conv2D layers with ELU activation
- Dropout for regularization
- Dense layers for classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class FEClassifier:
    """
    Fast Extraction (FE) classification model for blood detection
    """
    
    def __init__(self, 
                 input_shape: tuple,
                 num_classes: int = 8,
                 dropout_rate: float = 0.4,
                 learning_rate: float = 0.001):
        """
        Initialize FE classifier
        
        Args:
            input_shape: Shape of input patches (height, width, channels)
            num_classes: Number of classes to classify
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
        # Class names for blood detection
        self.class_names = {
            0: 'background',
            1: 'blood', 
            2: 'ketchup',
            3: 'artificial_blood',
            4: 'beetroot_juice', 
            5: 'poster_paint',
            6: 'tomato_concentrate',
            7: 'acrylic_paint'
        }
        
    def build_model(self) -> keras.Model:
        """
        Build the FE classification model architecture
        
        Returns:
            Compiled Keras model
        """
        logger.info("Building FE classification model...")
        
        model = models.Sequential([
            # First Conv2D layer
            layers.Conv2D(50, (5, 5), 
                         activation='elu',
                         input_shape=self.input_shape,
                         padding='same',
                         name='conv2d_1'),
            
            # Second Conv2D layer
            layers.Conv2D(100, (5, 5),
                         activation='elu', 
                         padding='same',
                         name='conv2d_2'),
            
            # Flatten for dense layers
            layers.Flatten(),
            
            # Dropout for regularization
            layers.Dropout(self.dropout_rate, name='dropout'),
            
            # Dense layer
            layers.Dense(100, activation='elu', name='dense_1'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        logger.info("Model architecture:")
        model.summary()
        
        return model
    
    def create_callbacks(self, patience: int = 10) -> list:
        """
        Create training callbacks
        
        Args:
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callback_list
    
    def prepare_data(self, 
                    patches: np.ndarray, 
                    labels: np.ndarray,
                    test_size: float = 0.2,
                    random_state: int = 42) -> tuple:
        """
        Prepare data for training
        
        Args:
            patches: Input patches (samples, height, width, channels)
            labels: Target labels (samples,)
            test_size: Fraction of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        # Remove background class (class 0) for training
        non_bg_mask = labels > 0
        patches_filtered = patches[non_bg_mask]
        labels_filtered = labels[non_bg_mask]
        
        # Adjust labels (subtract 1 to make them 0-indexed)
        labels_adjusted = labels_filtered - 1
        
        # Convert to categorical
        y_categorical = keras.utils.to_categorical(labels_adjusted, self.num_classes - 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            patches_filtered, y_categorical,
            test_size=test_size,
            random_state=random_state,
            stratify=labels_adjusted
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of classes: {self.num_classes - 1}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, 
             X_train: np.ndarray,
             y_train: np.ndarray, 
             X_val: np.ndarray,
             y_val: np.ndarray,
             epochs: int = 100,
             batch_size: int = 256,
             use_callbacks: bool = True) -> dict:
        """
        Train the FE classifier
        
        Args:
            X_train: Training patches
            y_train: Training labels
            X_val: Validation patches  
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            use_callbacks: Whether to use callbacks
            
        Returns:
            Training history
        """
        if self.model is None:
            # Update num_classes to exclude background
            self.num_classes = y_train.shape[1]
            self.build_model()
        
        logger.info("Starting training...")
        
        # Prepare callbacks
        callback_list = self.create_callbacks() if use_callbacks else []
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=1
        )
        
        self.history = history
        
        logger.info("Training completed!")
        
        return history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test patches
            y_test: Test labels
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info("Evaluating model...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        target_names = [self.class_names[i+1] for i in range(self.num_classes)]  # Exclude background
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_true,
            'prediction_probabilities': y_pred_proba
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        return results
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Training Loss')
        axes[1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray, save_path: str = None):
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix to plot
            save_path: Path to save the plot
        """
        # Get class names (excluding background)
        class_labels = [self.class_names[i+1] for i in range(self.num_classes)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def predict_image(self, image_patches: np.ndarray) -> np.ndarray:
        """
        Predict classes for image patches
        
        Args:
            image_patches: Input patches (samples, height, width, channels)
            
        Returns:
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = self.model.predict(image_patches)
        return predictions
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


def create_synthetic_data(num_samples: int = 1000, patch_size: int = 9, num_bands: int = 50) -> tuple:
    """
    Create synthetic data for testing
    
    Args:
        num_samples: Number of samples to generate
        patch_size: Size of spatial patches
        num_bands: Number of spectral bands
        
    Returns:
        Tuple of (patches, labels)
    """
    np.random.seed(42)
    
    # Generate synthetic patches
    patches = np.random.randn(num_samples, patch_size, patch_size, num_bands)
    
    # Generate synthetic labels (classes 1-7, excluding background)
    labels = np.random.randint(1, 8, size=num_samples)
    
    return patches, labels


def main():
    """
    Example usage of FE classifier
    """
    # Create synthetic data for testing
    patches, labels = create_synthetic_data(num_samples=2000, patch_size=9, num_bands=50)
    
    print(f"Synthetic data created:")
    print(f"Patches shape: {patches.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    
    # Initialize classifier
    classifier = FEClassifier(
        input_shape=(9, 9, 50),
        num_classes=8,  # Including background class
        dropout_rate=0.4,
        learning_rate=0.001
    )
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(patches, labels)
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        epochs=10,  # Reduced for testing
        batch_size=32
    )
    
    # Evaluate model
    results = classifier.evaluate(X_test, y_test)
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    
    print(f"\nFinal test accuracy: {results['accuracy']:.4f}")


if __name__ == "__main__":
    main()
