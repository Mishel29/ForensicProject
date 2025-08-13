"""
Enhancing Transformation Reduction (ETR) Method
===============================================

Implementation of the ETR preprocessing method from the paper:
"Enhancing forensic blood detection using hyperspectral imaging and advanced preprocessing techniques"

The ETR method consists of two main steps:
1. Dimension reduction using enhanced covariance matrix
2. Morphological dilation for pixel position correction
"""

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class ETRPreprocessor:
    """
    Enhancing Transformation Reduction (ETR) preprocessing method
    """
    
    def __init__(self, n_components: int = 50, enhancement_factor: float = 0.1):
        """
        Initialize ETR preprocessor
        
        Args:
            n_components: Number of components to keep after dimension reduction
            enhancement_factor: Enhancement factor for covariance matrix (ε)
        """
        self.n_components = n_components
        self.enhancement_factor = enhancement_factor
        self.weight_matrix = None
        self.mean_data = None
        self.transformed_shape = None
        
    def _enhance_covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Enhance covariance matrix by subtracting correlation matrix
        
        Args:
            data: Input data (samples x features)
            
        Returns:
            Enhanced covariance matrix
        """
        # Calculate covariance matrix
        covariance_matrix = np.cov(data.T)
        
        # Calculate correlation matrix  
        correlation_matrix = np.corrcoef(data.T)
        
        # Enhance covariance matrix: C = Cov(X) - ε * Corr(X)
        enhanced_covariance = covariance_matrix - self.enhancement_factor * correlation_matrix
        
        return enhanced_covariance
    
    def _create_marker_image(self, data: np.ndarray) -> np.ndarray:
        """
        Create marker image by taking mean of each instance
        
        Args:
            data: Transformed data (height x width x features)
            
        Returns:
            Marker image (height x width)
        """
        return np.mean(data, axis=2)
    
    def _morphological_dilation(self, data: np.ndarray, marker: np.ndarray, n_iterations: int = 1) -> np.ndarray:
        """
        Apply morphological dilation to correct pixel positions
        
        Args:
            data: Input data (height x width x features)
            marker: Marker image (height x width) 
            n_iterations: Number of dilation iterations
            
        Returns:
            Dilated data
        """
        h, w, d = data.shape
        dilated_data = np.zeros_like(data)
        
        # Create structuring element for dilation
        struct_element = ndimage.generate_binary_structure(2, 1)
        
        for i in range(d):
            # Get current band
            current_band = data[:, :, i]
            
            # Apply morphological dilation
            dilated_band = current_band.copy()
            for _ in range(n_iterations):
                dilated_band = ndimage.binary_dilation(
                    dilated_band > np.mean(dilated_band), 
                    structure=struct_element
                ).astype(float)
                dilated_band = np.where(dilated_band, current_band, np.minimum(current_band, marker))
            
            dilated_data[:, :, i] = dilated_band
        
        return dilated_data
    
    def _apply_gaussian_distribution(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian distribution normalization to the data
        
        Args:
            data: Input data
            
        Returns:
            Normalized data
        """
        # Reshape to 2D for normalization
        original_shape = data.shape
        data_2d = data.reshape(-1, data.shape[-1])
        
        # Apply standardization (Gaussian normalization)
        mean = np.mean(data_2d, axis=0)
        std = np.std(data_2d, axis=0)
        
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        
        normalized_data = (data_2d - mean) / std
        
        return normalized_data.reshape(original_shape)
    
    def fit(self, data: np.ndarray) -> 'ETRPreprocessor':
        """
        Fit the ETR preprocessor to the data
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            Fitted preprocessor
        """
        logger.info("Fitting ETR preprocessor...")
        
        h, w, z = data.shape
        self.transformed_shape = (h, w)
        
        # Reshape data to 2D (pixels x bands)
        data_2d = data.reshape(h * w, z)
        
        # Step 1: Dimension reduction with enhanced covariance matrix
        logger.info("Step 1: Computing enhanced covariance matrix...")
        
        # Calculate enhanced covariance matrix
        enhanced_cov = self._enhance_covariance_matrix(data_2d)
        
        # Perform eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(enhanced_cov)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select top eigenvectors as weight matrix
        self.weight_matrix = eigenvectors[:, :self.n_components]
        
        # Store mean for later use
        self.mean_data = np.mean(data_2d, axis=0)
        
        logger.info(f"Dimension reduced from {z} to {self.n_components} bands")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted ETR preprocessor
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            Transformed data (height x width x n_components)
        """
        if self.weight_matrix is None:
            raise ValueError("ETR preprocessor must be fitted before transform")
        
        logger.info("Transforming data with ETR...")
        
        h, w, z = data.shape
        
        # Reshape to 2D
        data_2d = data.reshape(h * w, z)
        
        # Step 1: Apply dimension reduction
        logger.info("Step 1: Applying dimension reduction...")
        transformed_2d = np.dot(data_2d, self.weight_matrix)
        
        # Reshape back to 3D
        transformed_3d = transformed_2d.reshape(h, w, self.n_components)
        
        # Step 2: Morphological processing
        logger.info("Step 2: Applying morphological dilation...")
        
        # Create marker image
        marker = self._create_marker_image(transformed_3d)
        
        # Apply morphological dilation
        dilated_data = self._morphological_dilation(transformed_3d, marker)
        
        # Step 3: Apply Gaussian distribution normalization
        logger.info("Step 3: Applying Gaussian normalization...")
        final_data = self._apply_gaussian_distribution(dilated_data)
        
        logger.info(f"ETR transformation completed. Output shape: {final_data.shape}")
        
        return final_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the preprocessor and transform data in one step
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)


class PCAPreprocessor:
    """
    Standard PCA preprocessing for comparison with ETR
    """
    
    def __init__(self, n_components: int = 50):
        """
        Initialize PCA preprocessor
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.transformed_shape = None
        
    def fit(self, data: np.ndarray) -> 'PCAPreprocessor':
        """
        Fit PCA to the data
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            Fitted preprocessor
        """
        logger.info("Fitting PCA preprocessor...")
        
        h, w, z = data.shape
        self.transformed_shape = (h, w)
        
        # Reshape to 2D
        data_2d = data.reshape(h * w, z)
        
        # Fit PCA
        self.pca.fit(data_2d)
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
        logger.info(f"Total explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            PCA transformed data
        """
        h, w, z = data.shape
        
        # Reshape to 2D
        data_2d = data.reshape(h * w, z)
        
        # Transform with PCA
        transformed_2d = self.pca.transform(data_2d)
        
        # Reshape back to 3D
        transformed_3d = transformed_2d.reshape(h, w, self.n_components)
        
        logger.info(f"PCA transformation completed. Output shape: {transformed_3d.shape}")
        
        return transformed_3d
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step
        
        Args:
            data: Input hyperspectral data (height x width x bands)
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)


def compare_preprocessing_methods(data: np.ndarray, n_components: int = 50):
    """
    Compare ETR and PCA preprocessing methods
    
    Args:
        data: Input hyperspectral data
        n_components: Number of components for both methods
    """
    import matplotlib.pyplot as plt
    
    logger.info("Comparing ETR and PCA preprocessing methods...")
    
    # Apply ETR
    etr = ETRPreprocessor(n_components=n_components)
    etr_data = etr.fit_transform(data)
    
    # Apply PCA
    pca = PCAPreprocessor(n_components=n_components)
    pca_data = pca.fit_transform(data)
    
    # Visualize first component of each method
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original data (first band)
    axes[0, 0].imshow(data[:, :, 0], cmap='viridis')
    axes[0, 0].set_title('Original Data (Band 0)')
    axes[0, 0].axis('off')
    
    # ETR first component
    axes[0, 1].imshow(etr_data[:, :, 0], cmap='viridis')
    axes[0, 1].set_title('ETR (Component 0)')
    axes[0, 1].axis('off')
    
    # PCA first component
    axes[1, 0].imshow(pca_data[:, :, 0], cmap='viridis')
    axes[1, 0].set_title('PCA (Component 0)')
    axes[1, 0].axis('off')
    
    # Difference between ETR and PCA
    diff = np.abs(etr_data[:, :, 0] - pca_data[:, :, 0])
    axes[1, 1].imshow(diff, cmap='hot')
    axes[1, 1].set_title('Absolute Difference (ETR - PCA)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\nPreprocessing comparison:")
    print(f"ETR data range: [{etr_data.min():.3f}, {etr_data.max():.3f}]")
    print(f"PCA data range: [{pca_data.min():.3f}, {pca_data.max():.3f}]")
    print(f"ETR std: {np.std(etr_data):.3f}")
    print(f"PCA std: {np.std(pca_data):.3f}")


def main():
    """
    Example usage of ETR preprocessor
    """
    # Create synthetic hyperspectral data for testing
    np.random.seed(42)
    h, w, bands = 100, 100, 128
    synthetic_data = np.random.randn(h, w, bands)
    
    print(f"Testing ETR with synthetic data: {synthetic_data.shape}")
    
    # Test ETR
    etr = ETRPreprocessor(n_components=50)
    transformed_data = etr.fit_transform(synthetic_data)
    
    print(f"Transformed data shape: {transformed_data.shape}")
    print(f"Data range: [{transformed_data.min():.3f}, {transformed_data.max():.3f}]")
    
    # Compare with PCA
    compare_preprocessing_methods(synthetic_data)


if __name__ == "__main__":
    main()
