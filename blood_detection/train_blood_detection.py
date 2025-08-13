"""
Blood Detection Training Script
===============================

Main training script for the Fast Extraction (FE) framework for blood detection
using hyperspectral imaging and the HyperBlood dataset.

This script implements the complete pipeline from the paper:
"Enhancing forensic blood detection using hyperspectral imaging and advanced preprocessing techniques"
"""

import os
import sys
import numpy as np
import argparse
import logging
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules (will show import errors until packages are installed)
try:
    from utils.hyperblood_loader import HyperBloodLoader
    from src.etr_preprocessing import ETRPreprocessor, PCAPreprocessor
    from models.fe_classifier import FEClassifier
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages first")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('blood_detection_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BloodDetectionTrainer:
    """
    Main trainer class for blood detection using FE framework
    """
    
    def __init__(self, 
                 dataset_path: str,
                 results_dir: str = "./results",
                 preprocessing_method: str = "etr",
                 n_components: int = 50,
                 patch_size: int = 9):
        """
        Initialize the blood detection trainer
        
        Args:
            dataset_path: Path to HyperBlood dataset
            results_dir: Directory to save results
            preprocessing_method: "etr" or "pca"
            n_components: Number of components for dimensionality reduction
            patch_size: Size of spatial patches
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.preprocessing_method = preprocessing_method
        self.n_components = n_components
        self.patch_size = patch_size
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = HyperBloodLoader(dataset_path)
        self.preprocessor = None
        self.classifier = None
        
        # Training configuration
        self.train_images = ['E_1', 'E_21']  # E scenes for training
        self.test_images = ['F_1', 'F_1a']   # F scenes for testing
        
        logger.info(f"BloodDetectionTrainer initialized")
        logger.info(f"Dataset path: {dataset_path}")
        logger.info(f"Preprocessing method: {preprocessing_method}")
        logger.info(f"Components: {n_components}")
        logger.info(f"Patch size: {patch_size}")
    
    def setup_preprocessor(self):
        """
        Setup the preprocessing method (ETR or PCA)
        """
        logger.info(f"Setting up {self.preprocessing_method.upper()} preprocessor...")
        
        if self.preprocessing_method.lower() == "etr":
            self.preprocessor = ETRPreprocessor(
                n_components=self.n_components,
                enhancement_factor=0.1
            )
        elif self.preprocessing_method.lower() == "pca":
            self.preprocessor = PCAPreprocessor(
                n_components=self.n_components
            )
        else:
            raise ValueError(f"Unknown preprocessing method: {self.preprocessing_method}")
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the hyperspectral data
        
        Returns:
            Tuple of (train_patches, train_labels, test_patches, test_labels)
        """
        logger.info("Loading and preprocessing data...")
        
        # Load training data
        train_data_list = []
        for image_name in self.train_images:
            logger.info(f"Loading training image: {image_name}")
            data, wavelengths = self.data_loader.load_hyperspectral_data(image_name)
            train_data_list.append(data)
        
        # Fit preprocessor on first training image
        logger.info("Fitting preprocessor...")
        self.preprocessor.fit(train_data_list[0])
        
        # Process all training images
        train_patches_list = []
        train_labels_list = []
        
        for i, image_name in enumerate(self.train_images):
            # Preprocess data
            processed_data = self.preprocessor.transform(train_data_list[i])
            
            # Load annotations
            annotation = self.data_loader.load_annotations(image_name)
            
            # Create patches
            patches, labels = self.data_loader.create_patches(
                processed_data, annotation, self.patch_size
            )
            
            train_patches_list.append(patches)
            train_labels_list.append(labels)
            
            logger.info(f"Processed {image_name}: {len(patches)} patches")
        
        # Combine training data
        train_patches = np.concatenate(train_patches_list, axis=0)
        train_labels = np.concatenate(train_labels_list, axis=0)
        
        # Process test data
        test_patches_list = []
        test_labels_list = []
        
        for image_name in self.test_images:
            logger.info(f"Loading test image: {image_name}")
            
            # Load and preprocess
            data, wavelengths = self.data_loader.load_hyperspectral_data(image_name)
            processed_data = self.preprocessor.transform(data)
            
            # Load annotations
            annotation = self.data_loader.load_annotations(image_name)
            
            # Create patches
            patches, labels = self.data_loader.create_patches(
                processed_data, annotation, self.patch_size
            )
            
            test_patches_list.append(patches)
            test_labels_list.append(labels)
            
            logger.info(f"Processed {image_name}: {len(patches)} patches")
        
        # Combine test data
        test_patches = np.concatenate(test_patches_list, axis=0)
        test_labels = np.concatenate(test_labels_list, axis=0)
        
        logger.info(f"Total training patches: {len(train_patches)}")
        logger.info(f"Total test patches: {len(test_patches)}")
        
        return train_patches, train_labels, test_patches, test_labels
    
    def train_classifier(self, train_patches, train_labels, test_patches, test_labels):
        """
        Train the FE classifier
        
        Args:
            train_patches: Training patches
            train_labels: Training labels
            test_patches: Test patches
            test_labels: Test labels
            
        Returns:
            Training results dictionary
        """
        logger.info("Training FE classifier...")
        
        # Initialize classifier
        input_shape = (self.patch_size, self.patch_size, self.n_components)
        self.classifier = FEClassifier(
            input_shape=input_shape,
            num_classes=8,  # 7 classes + background
            dropout_rate=0.4,
            learning_rate=0.001
        )
        
        # Prepare data for training
        X_train, X_test, y_train, y_test = self.classifier.prepare_data(
            train_patches, train_labels, test_size=0.2
        )
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model
        history = self.classifier.train(
            X_train, y_train,
            X_val, y_val,
            epochs=100,
            batch_size=256,
            use_callbacks=True
        )
        
        # Evaluate on test set
        test_results = self.classifier.evaluate(X_test, y_test)
        
        # Prepare independent test data
        X_independent, _, y_independent, _ = self.classifier.prepare_data(
            test_patches, test_labels, test_size=1.0  # Use all as test
        )
        
        # Evaluate on independent test set
        independent_results = self.classifier.evaluate(X_independent, y_independent)
        
        results = {
            'training_history': history,
            'test_results': test_results,
            'independent_test_results': independent_results,
            'training_images': self.train_images,
            'test_images': self.test_images,
            'preprocessing_method': self.preprocessing_method,
            'n_components': self.n_components,
            'patch_size': self.patch_size
        }
        
        return results
    
    def save_results(self, results, timestamp):
        """
        Save training results
        
        Args:
            results: Results dictionary
            timestamp: Timestamp string
        """
        logger.info("Saving results...")
        
        # Save model
        model_path = os.path.join(self.results_dir, f"fe_model_{timestamp}.h5")
        self.classifier.save_model(model_path)
        
        # Save training plots
        history_plot_path = os.path.join(self.results_dir, f"training_history_{timestamp}.png")
        self.classifier.plot_training_history(history_plot_path)
        
        # Save confusion matrices
        cm_test_path = os.path.join(self.results_dir, f"confusion_matrix_test_{timestamp}.png")
        self.classifier.plot_confusion_matrix(results['test_results']['confusion_matrix'], cm_test_path)
        
        cm_independent_path = os.path.join(self.results_dir, f"confusion_matrix_independent_{timestamp}.png")
        self.classifier.plot_confusion_matrix(results['independent_test_results']['confusion_matrix'], cm_independent_path)
        
        # Save numerical results
        results_summary = {
            'test_accuracy': float(results['test_results']['accuracy']),
            'independent_test_accuracy': float(results['independent_test_results']['accuracy']),
            'training_images': results['training_images'],
            'test_images': results['test_images'],
            'preprocessing_method': results['preprocessing_method'],
            'n_components': results['n_components'],
            'patch_size': results['patch_size'],
            'timestamp': timestamp
        }
        
        results_path = os.path.join(self.results_dir, f"results_summary_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        logger.info(f"Results saved with timestamp: {timestamp}")
        
        return results_summary
    
    def run_experiment(self):
        """
        Run the complete blood detection experiment
        
        Returns:
            Results summary
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting blood detection experiment - {timestamp}")
        
        try:
            # Setup preprocessor
            self.setup_preprocessor()
            
            # Load and preprocess data
            train_patches, train_labels, test_patches, test_labels = self.load_and_preprocess_data()
            
            # Train classifier
            results = self.train_classifier(train_patches, train_labels, test_patches, test_labels)
            
            # Save results
            results_summary = self.save_results(results, timestamp)
            
            logger.info("Experiment completed successfully!")
            logger.info(f"Test accuracy: {results_summary['test_accuracy']:.4f}")
            logger.info(f"Independent test accuracy: {results_summary['independent_test_accuracy']:.4f}")
            
            return results_summary
            
        except Exception as e:
            logger.error(f"Experiment failed: {str(e)}")
            raise


def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Blood Detection Training Script')
    parser.add_argument('--dataset_path', type=str, 
                       default='../data/raw/HyperBlood/',
                       help='Path to HyperBlood dataset')
    parser.add_argument('--results_dir', type=str,
                       default='./results',
                       help='Directory to save results')
    parser.add_argument('--preprocessing', type=str,
                       choices=['etr', 'pca'], default='etr',
                       help='Preprocessing method')
    parser.add_argument('--n_components', type=int, default=50,
                       help='Number of components for dimensionality reduction')
    parser.add_argument('--patch_size', type=int, default=9,
                       help='Size of spatial patches')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BloodDetectionTrainer(
        dataset_path=args.dataset_path,
        results_dir=args.results_dir,
        preprocessing_method=args.preprocessing,
        n_components=args.n_components,
        patch_size=args.patch_size
    )
    
    # Run experiment
    try:
        results = trainer.run_experiment()
        print("\n" + "="*50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Independent Test Accuracy: {results['independent_test_accuracy']:.4f}")
        print(f"Results saved in: {args.results_dir}")
        
    except Exception as e:
        print(f"\nExperiment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
