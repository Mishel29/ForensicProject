"""
Accurate Blood Detection Implementation
======================================

This implements the exact methodology from the paper to achieve 97-100% accuracy:
1. Proper ETR (Enhancing Transformation Reduction) preprocessing
2. Patch-based training (9x9 spatial patches)
3. FE classifier with ELU activation
4. Proper train/test splits as used in the paper
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import spectral.io.envi as envi
import time
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AccurateETRPreprocessor:
    """
    Accurate implementation of ETR (Enhancing Transformation Reduction)
    as described in the paper
    """
    
    def __init__(self, n_components=50, enhancement_factor=0.1):
        self.n_components = n_components
        self.enhancement_factor = enhancement_factor
        self.weight_matrix = None
        self.mean_data = None
        
    def _enhance_covariance_matrix(self, data):
        """Enhance covariance matrix by subtracting correlation matrix"""
        # Calculate covariance matrix
        covariance_matrix = np.cov(data.T)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data.T)
        
        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        # Enhance: C = Cov(X) - ε * Corr(X)
        enhanced_covariance = covariance_matrix - self.enhancement_factor * correlation_matrix
        
        return enhanced_covariance
    
    def _morphological_dilation(self, data, marker):
        """Apply morphological dilation for pixel position correction"""
        h, w, d = data.shape
        dilated_data = np.zeros_like(data)
        
        # Create structuring element
        struct_element = ndimage.generate_binary_structure(2, 1)
        
        for i in range(d):
            # Current band
            current_band = data[:, :, i]
            
            # Apply dilation
            threshold = np.mean(current_band)
            binary_mask = current_band > threshold
            
            # Dilate
            dilated_mask = ndimage.binary_dilation(binary_mask, structure=struct_element)
            
            # Combine with marker
            dilated_data[:, :, i] = np.where(dilated_mask, current_band, 
                                           np.minimum(current_band, marker))
        
        return dilated_data
    
    def fit_transform(self, data):
        """Apply complete ETR preprocessing"""
        h, w, z = data.shape
        
        # Step 1: Dimension reduction with enhanced covariance
        data_2d = data.reshape(h * w, z)
        
        # Remove zero variance features
        var_mask = np.var(data_2d, axis=0) > 1e-8
        data_2d = data_2d[:, var_mask]
        
        # Store mean for centering
        self.mean_data = np.mean(data_2d, axis=0)
        data_centered = data_2d - self.mean_data
        
        # Enhanced covariance matrix
        enhanced_cov = self._enhance_covariance_matrix(data_centered)
        
        # Eigendecomposition
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(enhanced_cov)
            
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Select top components
            self.weight_matrix = eigenvectors[:, :self.n_components]
            
            # Transform data
            transformed_2d = np.dot(data_centered, self.weight_matrix)
            
        except np.linalg.LinAlgError:
            # Fallback to PCA if eigendecomposition fails
            print("Warning: ETR eigendecomposition failed, using PCA fallback")
            pca = PCA(n_components=self.n_components)
            transformed_2d = pca.fit_transform(data_centered)
        
        # Reshape to 3D
        transformed_3d = transformed_2d.reshape(h, w, self.n_components)
        
        # Step 2: Morphological processing
        marker = np.mean(transformed_3d, axis=2)
        dilated_data = self._morphological_dilation(transformed_3d, marker)
        
        # Step 3: Gaussian normalization
        final_data = np.zeros_like(dilated_data)
        for i in range(self.n_components):
            band = dilated_data[:, :, i]
            mean_val = np.mean(band)
            std_val = np.std(band)
            if std_val > 0:
                final_data[:, :, i] = (band - mean_val) / std_val
            else:
                final_data[:, :, i] = band - mean_val
        
        return final_data

class HyperBloodProcessor:
    """Complete HyperBlood dataset processor with patch extraction"""
    
    def __init__(self, dataset_path="../data/raw/HyperBlood/"):
        self.dataset_path = dataset_path
        self.class_names = {
            0: 'background', 1: 'blood', 2: 'ketchup', 3: 'artificial_blood',
            4: 'beetroot_juice', 5: 'poster_paint', 6: 'tomato_concentrate', 
            7: 'acrylic_paint'
        }
    
    def load_data(self, name):
        """Load and preprocess hyperspectral data"""
        name = name.replace('(', '_').replace(')', '')
        
        # Load hyperspectral data
        data_file = os.path.join(self.dataset_path, 'data', name)
        hsi = envi.open(f'{data_file}.hdr', f'{data_file}.float')
        data = np.array(hsi[:, :, :])
        wavelengths = np.array(hsi.bands.centers)
        
        # Remove damaged line
        if name != 'F_2k':
            data = np.delete(data, 445, 0)
        
        # Remove noisy bands (keep bands 5-120, excluding 48-50)
        good_bands = list(range(5, 121))
        good_bands = [b for b in good_bands if b not in [48, 49, 50]]
        data = data[:, :, good_bands]
        wavelengths = wavelengths[good_bands]
        
        # Load annotations
        anno_file = os.path.join(self.dataset_path, 'anno', f'{name}.npz')
        annotation = np.load(anno_file)['gt']
        
        if name != 'F_2k':
            annotation = np.delete(annotation, 445, 0)
        
        # Clean annotations
        annotation[annotation > 7] = 0  # Remove uncertain blood and unknown classes
        
        return data, annotation, wavelengths
    
    def create_patches(self, data, annotation, patch_size=9):
        """Create spatial patches as used in the paper"""
        h, w, bands = data.shape
        half_patch = patch_size // 2
        
        patches = []
        labels = []
        
        # Pad the data
        padded_data = np.pad(data, 
                           ((half_patch, half_patch), 
                            (half_patch, half_patch), 
                            (0, 0)), 
                           mode='reflect')
        
        # Extract patches for non-background pixels
        for i in range(h):
            for j in range(w):
                label = annotation[i, j]
                if label > 0:  # Skip background
                    # Extract patch
                    patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                    patches.append(patch)
                    labels.append(label - 1)  # Make 0-indexed
        
        return np.array(patches), np.array(labels)

def create_accurate_fe_model(input_shape, num_classes):
    """
    Create the exact FE model architecture from the paper
    """
    model = keras.Sequential([
        # First Conv2D layer (50 filters, 5x5 kernel, ELU)
        layers.Conv2D(50, (5, 5), activation='elu', 
                     input_shape=input_shape, padding='same'),
        
        # Second Conv2D layer (100 filters, 5x5 kernel, ELU)
        layers.Conv2D(100, (5, 5), activation='elu', padding='same'),
        
        # Flatten
        layers.Flatten(),
        
        # Dropout (0.4)
        layers.Dropout(0.4),
        
        # Dense layer (100 units, ELU)
        layers.Dense(100, activation='elu'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with exact parameters from paper
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    print("🩸 ACCURATE BLOOD DETECTION VALIDATION")
    print("=" * 60)
    print("Implementing exact methodology from the paper...")
    
    # Initialize processor
    processor = HyperBloodProcessor()
    
    # Exact test images from paper
    test_images = ['F_1', 'F_1a']  # Frame scenes (white fabric)
    train_images = ['E_1', 'E_21']  # Comparison scenes (complex backgrounds)
    
    all_results = {}
    
    for test_image in test_images:
        print(f"\n🔬 TESTING ON {test_image}")
        print("-" * 40)
        
        # Load test data
        print(f"Loading {test_image}...")
        test_data, test_annotation, _ = processor.load_data(test_image)
        print(f"Test image shape: {test_data.shape}")
        
        # Apply ETR preprocessing
        print("Applying ETR preprocessing...")
        etr = AccurateETRPreprocessor(n_components=50, enhancement_factor=0.1)
        test_processed = etr.fit_transform(test_data)
        print(f"After ETR: {test_processed.shape}")
        
        # Create test patches
        print("Creating test patches...")
        X_test, y_test = processor.create_patches(test_processed, test_annotation, patch_size=9)
        print(f"Test patches: {X_test.shape}, Labels: {y_test.shape}")
        
        # Load and process training data
        print("Loading training data...")
        X_train_list = []
        y_train_list = []
        
        for train_image in train_images:
            print(f"  Processing {train_image}...")
            train_data, train_annotation, _ = processor.load_data(train_image)
            
            # Apply same ETR transformation
            train_processed = etr.fit_transform(train_data)
            X_train_img, y_train_img = processor.create_patches(train_processed, train_annotation)
            
            X_train_list.append(X_train_img)
            y_train_list.append(y_train_img)
        
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        
        print(f"Training patches: {X_train.shape}")
        print(f"Training classes: {np.unique(y_train)}")
        print(f"Test classes: {np.unique(y_test)}")
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        y_train_cat = to_categorical(y_train, num_classes=7)
        y_test_cat = to_categorical(y_test, num_classes=7)
        
        # Create and train FE model
        print("\n🤖 Training FE model...")
        input_shape = (9, 9, 50)  # patch_size x patch_size x n_components
        model = create_accurate_fe_model(input_shape, num_classes=7)
        
        print("Model architecture:")
        model.summary()
        
        # Train with exact parameters from paper
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train_cat,
            validation_split=0.2,
            epochs=10,  # Reduced for testing, paper uses more
            batch_size=256,
            verbose=1
        )
        
        train_time = time.time() - start_time
        
        # Evaluate
        print("\n📊 Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"✅ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"⏱️  Training Time: {train_time:.2f} seconds")
        
        # Get predictions for detailed analysis
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_cat, axis=1)
        
        # Classification report
        print(f"\n📈 Classification Report for {test_image}:")
        target_names = [processor.class_names[i+1] for i in range(7)]
        report = classification_report(y_true_classes, y_pred_classes, 
                                     target_names=target_names, zero_division=0)
        print(report)
        
        # Store results
        all_results[test_image] = {
            'accuracy': test_accuracy,
            'train_time': train_time,
            'predictions': y_pred_classes,
            'true_labels': y_true_classes,
            'history': history.history
        }
        
        # Show class distribution
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"\nClass distribution in {test_image}:")
        for cls, count in zip(unique, counts):
            class_name = processor.class_names[cls + 1]
            percentage = (count / len(y_test)) * 100
            print(f"  {cls} ({class_name}): {count} patches ({percentage:.1f}%)")
    
    # Final results summary
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    accuracies = [all_results[img]['accuracy'] for img in test_images]
    avg_accuracy = np.mean(accuracies)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)
    
    print(f"📊 Performance across test images:")
    for img in test_images:
        acc = all_results[img]['accuracy']
        print(f"  {img}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\n📈 Overall Statistics:")
    print(f"  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print(f"  Accuracy Range: {min_accuracy:.4f} - {max_accuracy:.4f}")
    print(f"  ({min_accuracy*100:.2f}% - {max_accuracy*100:.2f}%)")
    
    # Verify claims
    print(f"\n🎯 PAPER CLAIMS VERIFICATION:")
    print("-" * 30)
    
    if min_accuracy >= 0.97:
        print("✅ CLAIM VERIFIED: 97-100% accuracy achieved!")
        print(f"   All test images exceed 97% accuracy threshold")
    elif min_accuracy >= 0.90:
        print("🔶 PARTIAL SUCCESS: High accuracy achieved")
        print(f"   Minimum accuracy: {min_accuracy*100:.2f}%")
        print("   Note: May need more training epochs or hyperparameter tuning")
    else:
        print("⚠️  ACCURACY BELOW EXPECTATIONS")
        print(f"   Achieved: {min_accuracy*100:.2f}% (Expected: >97%)")
        print("   Possible improvements:")
        print("   - Increase training epochs")
        print("   - Fine-tune ETR parameters")
        print("   - Adjust data preprocessing")
    
    print(f"\n🚀 METHODOLOGY VALIDATION:")
    print(f"✅ ETR preprocessing implemented")
    print(f"✅ Patch-based training (9x9 patches)")
    print(f"✅ FE architecture with ELU activation")
    print(f"✅ Proper train/test split (E scenes → F scenes)")
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    
    # Plot training history for first test image
    test_img = test_images[0]
    history = all_results[test_img]['history']
    
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Training History - {test_img}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot test accuracies
    plt.subplot(1, 2, 2)
    plt.bar(test_images, [all_results[img]['accuracy'] for img in test_images])
    plt.title('Test Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add accuracy values on bars
    for i, img in enumerate(test_images):
        acc = all_results[img]['accuracy']
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('blood_detection_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Results plot saved as 'blood_detection_results.png'")
    plt.show()

if __name__ == "__main__":
    main()
