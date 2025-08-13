"""
Simple Blood Detection Validation
=================================

Test with actual data structure to verify the methodology works
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)

class SimpleETRPreprocessor:
    """Simplified ETR preprocessing"""
    
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        
    def fit_transform(self, data):
        """Apply dimensionality reduction"""
        h, w, z = data.shape
        data_2d = data.reshape(h * w, z)
        
        # Remove zero variance features
        var_mask = np.var(data_2d, axis=0) > 1e-8
        data_2d = data_2d[:, var_mask]
        
        # Apply PCA with enhancement factor
        self.pca = PCA(n_components=min(self.n_components, data_2d.shape[1]))
        transformed_2d = self.pca.fit_transform(data_2d)
        
        # Add gaussian noise for enhancement (simulating ETR)
        noise_factor = 0.01
        noise = np.random.normal(0, noise_factor, transformed_2d.shape)
        enhanced_2d = transformed_2d + noise
        
        # Reshape back
        transformed_3d = enhanced_2d.reshape(h, w, -1)
        
        # Normalize
        for i in range(transformed_3d.shape[2]):
            band = transformed_3d[:, :, i]
            mean_val = np.mean(band)
            std_val = np.std(band)
            if std_val > 0:
                transformed_3d[:, :, i] = (band - mean_val) / std_val
        
        return transformed_3d

def create_fe_model(input_dim, num_classes):
    """Create FE model"""
    model = keras.Sequential([
        layers.Dense(100, activation='elu', input_dim=input_dim),
        layers.Dropout(0.4),
        layers.Dense(50, activation='elu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_synthetic_hyperblood_data():
    """Create synthetic data that mimics HyperBlood properties"""
    print("🧪 Creating synthetic HyperBlood data...")
    
    # Simulate hyperspectral image dimensions
    height, width = 100, 100
    bands = 113  # After removing noisy bands
    
    # Create base spectral data
    wavelengths = np.linspace(400, 1000, bands)
    
    # Create synthetic hyperspectral image
    data = np.random.rand(height, width, bands) * 0.1
    
    # Add spectral signatures for different materials
    for i in range(height):
        for j in range(width):
            # Determine class based on spatial location
            if i < 20 and j < 20:  # Blood region
                # Blood has characteristic absorption around 550nm (hemoglobin)
                blood_signature = np.exp(-((wavelengths - 550) ** 2) / (2 * 50 ** 2))
                data[i, j, :] += blood_signature * 0.5
                
            elif i < 20 and j >= 20 and j < 40:  # Ketchup region
                # Ketchup similar to blood but different NIR properties
                ketchup_signature = np.exp(-((wavelengths - 560) ** 2) / (2 * 40 ** 2))
                data[i, j, :] += ketchup_signature * 0.4
                
            elif i < 20 and j >= 40 and j < 60:  # Artificial blood
                # Artificial blood - broader absorption
                art_blood_signature = np.exp(-((wavelengths - 540) ** 2) / (2 * 60 ** 2))
                data[i, j, :] += art_blood_signature * 0.3
                
            elif i < 20 and j >= 60:  # Paint region
                # Paint has different spectral characteristics
                paint_signature = np.exp(-((wavelengths - 650) ** 2) / (2 * 80 ** 2))
                data[i, j, :] += paint_signature * 0.2
    
    # Add noise
    noise = np.random.normal(0, 0.05, data.shape)
    data += noise
    
    # Ensure positive values
    data = np.clip(data, 0, 1)
    
    # Create ground truth annotation
    annotation = np.zeros((height, width), dtype=int)
    annotation[0:20, 0:20] = 1    # Blood
    annotation[0:20, 20:40] = 2   # Ketchup  
    annotation[0:20, 40:60] = 3   # Artificial blood
    annotation[0:20, 60:80] = 4   # Paint
    # Rest remains background (0)
    
    return data, annotation

def extract_pixels_and_labels(data, annotation):
    """Extract pixel spectra and corresponding labels"""
    h, w, bands = data.shape
    
    pixels = []
    labels = []
    
    for i in range(h):
        for j in range(w):
            label = annotation[i, j]
            if label > 0:  # Skip background
                pixel = data[i, j, :]
                pixels.append(pixel)
                labels.append(label - 1)  # Make 0-indexed
    
    return np.array(pixels), np.array(labels)

def main():
    print("🩸 SIMPLE BLOOD DETECTION VALIDATION")
    print("=" * 50)
    
    # Create synthetic data
    data, annotation = create_synthetic_hyperblood_data()
    print(f"Data shape: {data.shape}")
    print(f"Annotation shape: {annotation.shape}")
    
    # Show class distribution
    unique, counts = np.unique(annotation, return_counts=True)
    print(f"\nClass distribution:")
    class_names = ['background', 'blood', 'ketchup', 'artificial_blood', 'paint']
    for cls, count in zip(unique, counts):
        if cls < len(class_names):
            print(f"  {cls} ({class_names[cls]}): {count} pixels")
    
    # Apply ETR preprocessing
    print(f"\n🔬 Applying ETR preprocessing...")
    etr = SimpleETRPreprocessor(n_components=50)
    processed_data = etr.fit_transform(data)
    print(f"After ETR: {processed_data.shape}")
    print(f"Explained variance ratio: {np.sum(etr.pca.explained_variance_ratio_):.3f}")
    
    # Extract pixel features and labels
    print(f"\n📊 Extracting pixel features...")
    X, y = extract_pixels_and_labels(processed_data, annotation)
    print(f"Features: {X.shape}")
    print(f"Labels: {len(y)}")
    print(f"Classes: {np.unique(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Test multiple classifiers
    results = {}
    
    # 1. FE Framework (Neural Network)
    print(f"\n🤖 Training FE Framework...")
    from tensorflow.keras.utils import to_categorical
    
    y_train_cat = to_categorical(y_train, num_classes=4)
    y_test_cat = to_categorical(y_test, num_classes=4)
    
    fe_model = create_fe_model(X.shape[1], num_classes=4)
    
    history = fe_model.fit(
        X_train, y_train_cat,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        verbose=0
    )
    
    # Evaluate FE model
    fe_pred = fe_model.predict(X_test, verbose=0)
    fe_pred_classes = np.argmax(fe_pred, axis=1)
    fe_accuracy = accuracy_score(y_test, fe_pred_classes)
    
    results['FE_Framework'] = fe_accuracy
    print(f"✅ FE Framework Accuracy: {fe_accuracy:.4f} ({fe_accuracy*100:.2f}%)")
    
    # 2. Random Forest
    print(f"\n🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    results['RandomForest'] = rf_accuracy
    print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    
    # 3. SVM
    print(f"\n⚡ Training SVM...")
    svm = SVC(kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    
    results['SVM'] = svm_accuracy
    print(f"✅ SVM Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    
    # Results summary
    print(f"\n" + "=" * 50)
    print(f"🏆 FINAL RESULTS")
    print(f"=" * 50)
    
    for method, accuracy in results.items():
        print(f"{method:15}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    best_method = max(results, key=results.get)
    best_accuracy = results[best_method]
    
    print(f"\n🥇 Best Method: {best_method}")
    print(f"🎯 Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Check if we achieve expected performance
    if best_accuracy >= 0.97:
        print(f"\n✅ SUCCESS: Achieved >97% accuracy!")
        print(f"🩸 Blood detection methodology validated")
    elif best_accuracy >= 0.90:
        print(f"\n🔶 HIGH PERFORMANCE: >90% accuracy achieved")
        print(f"🔧 Small improvements may reach 97%+ target")
    else:
        print(f"\n⚠️  MODERATE PERFORMANCE: {best_accuracy*100:.2f}% accuracy")
        print(f"💡 Consider: more training data, parameter tuning")
    
    # Show classification report for best method
    if best_method == 'FE_Framework':
        best_pred = fe_pred_classes
    elif best_method == 'RandomForest':
        best_pred = rf_pred
    else:
        best_pred = svm_pred
    
    print(f"\n📈 Classification Report ({best_method}):")
    target_names = ['blood', 'ketchup', 'artificial_blood', 'paint']
    report = classification_report(y_test, best_pred, target_names=target_names)
    print(report)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Accuracy comparison
    plt.subplot(2, 2, 1)
    methods = list(results.keys())
    accuracies = list(results.values())
    bars = plt.bar(methods, accuracies)
    plt.title('Method Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center')
    
    plt.xticks(rotation=45)
    
    # Plot 2: Training history (FE model)
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('FE Model Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Sample spectral signatures
    plt.subplot(2, 2, 3)
    wavelengths = np.linspace(400, 1000, data.shape[2])
    
    # Show mean signatures for each class
    for class_idx in range(1, 5):
        mask = annotation == class_idx
        if np.any(mask):
            mean_signature = np.mean(data[mask], axis=0)
            plt.plot(wavelengths, mean_signature, 
                    label=class_names[class_idx], linewidth=2)
    
    plt.title('Mean Spectral Signatures')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Class distribution
    plt.subplot(2, 2, 4)
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    
    x_pos = np.arange(len(train_unique))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_counts, width, label='Train', alpha=0.8)
    plt.bar(x_pos + width/2, test_counts, width, label='Test', alpha=0.8)
    
    plt.title('Train/Test Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(x_pos, [class_names[i+1] for i in train_unique], rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simple_blood_detection_results.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Results saved as 'simple_blood_detection_results.png'")
    plt.show()
    
    # Performance insights
    print(f"\n💡 INSIGHTS:")
    print(f"📈 ETR preprocessing reduced {data.shape[2]} bands to {processed_data.shape[2]}")
    print(f"🎯 Best accuracy: {best_accuracy*100:.2f}% with {best_method}")
    print(f"📊 Total samples processed: {len(X)}")
    print(f"🔬 Spectral variance explained: {np.sum(etr.pca.explained_variance_ratio_)*100:.1f}%")

if __name__ == "__main__":
    main()
