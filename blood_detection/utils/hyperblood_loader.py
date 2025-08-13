"""
HyperBlood Dataset Loader
=========================

This module provides functionality to load and preprocess the HyperBlood dataset
for forensic blood detection using hyperspectral imaging.

Based on the paper: "Enhancing forensic blood detection using hyperspectral imaging 
and advanced preprocessing techniques" by Dalal AL-Alimi and Mohammed A.A. Al-qaness

Classes:
0 - background
1 - blood
2 - ketchup  
3 - artificial blood
4 - beetroot juice
5 - poster paint
6 - tomato concentrate
7 - acrylic paint
8 - uncertain blood
"""

import os
import numpy as np
import spectral.io.envi as envi
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperBloodLoader:
    """
    Loader class for HyperBlood hyperspectral dataset
    """
    
    def __init__(self, dataset_path: str = "../data/raw/HyperBlood/"):
        """
        Initialize the HyperBlood dataset loader
        
        Args:
            dataset_path: Path to the HyperBlood dataset directory
        """
        self.dataset_path = dataset_path
        self.data_path = os.path.join(dataset_path, "data")
        self.anno_path = os.path.join(dataset_path, "anno")
        
        # Available images in the dataset
        self.available_images = [
            'A_1', 'B_1', 'C_1', 'D_1', 'E_1', 'E_7', 'E_21', 
            'F_1', 'F_1a', 'F_1s', 'F_2', 'F_2k', 'F_7', 'F_21'
        ]
        
        # Class names mapping
        self.class_names = {
            0: 'background',
            1: 'blood',
            2: 'ketchup',
            3: 'artificial_blood', 
            4: 'beetroot_juice',
            5: 'poster_paint',
            6: 'tomato_concentrate',
            7: 'acrylic_paint',
            8: 'uncertain_blood'
        }
        
        # Noisy bands to remove (indices)
        self.noisy_bands = np.array([0, 1, 2, 3, 4, 48, 49, 50, 121, 122, 123, 124, 125, 126, 127])
        
        logger.info(f"HyperBlood loader initialized with dataset path: {dataset_path}")
        
    def convert_name(self, name: str) -> str:
        """
        Convert image name from display format to file format
        
        Args:
            name: Image name (e.g., 'E(1)' or 'E_1')
            
        Returns:
            Converted filename format
        """
        name = name.replace('(', '_').replace(')', '')
        return name
    
    def get_good_indices(self, name: str = None) -> np.ndarray:
        """
        Get indices of spectral bands that are not noisy
        
        Args:
            name: Image name
            
        Returns:
            Array of good band indices
        """
        name = self.convert_name(name) if name else None
        
        if name != 'F_2k':
            indices = np.arange(128)
            indices = indices[5:-7]  # Remove first 5 and last 7 bands
        else:
            indices = np.arange(116)
            
        # Remove specific noisy bands
        indices = np.delete(indices, [43, 44, 45])
        return indices
    
    def load_hyperspectral_data(self, 
                               name: str, 
                               remove_bands: bool = True, 
                               clean: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load hyperspectral data from ENVI format
        
        Args:
            name: Image name
            remove_bands: Whether to remove noisy bands
            clean: Whether to remove damaged sensor line
            
        Returns:
            Tuple of (data, wavelengths)
        """
        name = self.convert_name(name)
        filename = os.path.join(self.data_path, name)
        
        try:
            hsimage = envi.open(f'{filename}.hdr', f'{filename}.float')
            wavelengths = np.asarray(hsimage.bands.centers)
            data = np.asarray(hsimage[:, :, :], dtype=np.float32)
            
            # Remove damaged sensor line (except for F_2k)
            if clean and name != 'F_2k':
                data = np.delete(data, 445, 0)
            
            if not remove_bands:
                return data, wavelengths
                
            good_indices = self.get_good_indices(name)
            return data[:, :, good_indices], wavelengths[good_indices]
            
        except Exception as e:
            logger.error(f"Error loading data for {name}: {str(e)}")
            raise
    
    def load_annotations(self, 
                        name: str, 
                        remove_uncertain_blood: bool = True, 
                        clean: bool = True) -> np.ndarray:
        """
        Load ground truth annotations
        
        Args:
            name: Image name
            remove_uncertain_blood: Whether to remove uncertain blood class
            clean: Whether to remove damaged sensor line
            
        Returns:
            2D annotation array
        """
        name = self.convert_name(name)
        filename = os.path.join(self.anno_path, f"{name}.npz")
        
        try:
            annotation = np.load(filename)['gt']
            
            # Remove damaged sensor line (except for F_2k)
            if clean and name != 'F_2k':
                annotation = np.delete(annotation, 445, 0)
            
            # Remove uncertain blood and technical classes
            if remove_uncertain_blood:
                annotation[annotation > 7] = 0
            else:
                annotation[annotation > 8] = 0
                
            return annotation
            
        except Exception as e:
            logger.error(f"Error loading annotations for {name}: {str(e)}")
            raise
    
    def get_rgb_visualization(self, 
                            data: np.ndarray, 
                            wavelengths: np.ndarray, 
                            gamma: float = 0.7, 
                            rgb_bands: List[int] = [600, 550, 450]) -> np.ndarray:
        """
        Create RGB visualization of hyperspectral data
        
        Args:
            data: Hyperspectral data cube
            wavelengths: Band wavelengths
            gamma: Gamma correction value
            rgb_bands: Wavelengths for RGB channels
            
        Returns:
            RGB image array
        """
        assert data.shape[2] == len(wavelengths)
        
        max_data = np.max(data)
        rgb_indices = [np.argmin(np.abs(wavelengths - band)) for band in rgb_bands]
        rgb_image = data[:, :, rgb_indices].copy() / max_data
        
        if gamma != 1.0:
            for i in range(3):
                rgb_image[:, :, i] = np.power(rgb_image[:, :, i], gamma)
        
        return rgb_image
    
    def get_class_statistics(self, annotation: np.ndarray) -> dict:
        """
        Get statistics about class distribution in annotation
        
        Args:
            annotation: Ground truth annotation array
            
        Returns:
            Dictionary with class statistics
        """
        unique_classes, counts = np.unique(annotation, return_counts=True)
        total_pixels = annotation.size
        
        stats = {}
        for cls, count in zip(unique_classes, counts):
            stats[cls] = {
                'name': self.class_names.get(cls, 'unknown'),
                'count': count,
                'percentage': (count / total_pixels) * 100
            }
        
        return stats
    
    def extract_pixels_by_class(self, 
                               data: np.ndarray, 
                               annotation: np.ndarray, 
                               target_class: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pixels belonging to a specific class
        
        Args:
            data: Hyperspectral data
            annotation: Ground truth annotation
            target_class: Class to extract
            
        Returns:
            Tuple of (pixel_spectra, pixel_coordinates)
        """
        class_mask = annotation == target_class
        class_pixels = data[class_mask]
        class_coords = np.where(class_mask)
        
        return class_pixels, np.column_stack(class_coords)
    
    def create_patches(self, 
                      data: np.ndarray, 
                      annotation: np.ndarray, 
                      patch_size: int = 9, 
                      stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial patches from hyperspectral data
        
        Args:
            data: Hyperspectral data cube
            annotation: Ground truth annotation
            patch_size: Size of spatial patches
            stride: Stride for patch extraction
            
        Returns:
            Tuple of (patches, labels)
        """
        h, w, bands = data.shape
        half_patch = patch_size // 2
        
        # Pad data and annotation
        padded_data = np.pad(data, 
                           ((half_patch, half_patch), 
                            (half_patch, half_patch), 
                            (0, 0)), 
                           mode='constant', constant_values=0)
        
        patches = []
        labels = []
        
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # Extract patch
                patch = padded_data[i:i+patch_size, j:j+patch_size, :]
                label = annotation[i, j]
                
                # Only include non-background patches
                if label > 0:
                    patches.append(patch)
                    labels.append(label)
        
        return np.array(patches), np.array(labels)
    
    def load_dataset_split(self, 
                          image_names: List[str], 
                          patch_size: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load multiple images and create a combined dataset
        
        Args:
            image_names: List of image names to load
            patch_size: Size of spatial patches
            
        Returns:
            Tuple of (all_patches, all_labels)
        """
        all_patches = []
        all_labels = []
        
        for name in image_names:
            logger.info(f"Loading image: {name}")
            
            # Load data and annotations
            data, wavelengths = self.load_hyperspectral_data(name)
            annotation = self.load_annotations(name)
            
            # Create patches
            patches, labels = self.create_patches(data, annotation, patch_size)
            
            all_patches.append(patches)
            all_labels.append(labels)
            
            logger.info(f"Extracted {len(patches)} patches from {name}")
        
        # Combine all patches
        combined_patches = np.concatenate(all_patches, axis=0)
        combined_labels = np.concatenate(all_labels, axis=0)
        
        logger.info(f"Total patches: {len(combined_patches)}")
        
        return combined_patches, combined_labels
    
    def visualize_image(self, name: str, save_path: Optional[str] = None):
        """
        Visualize hyperspectral image and its annotation
        
        Args:
            name: Image name
            save_path: Optional path to save the visualization
        """
        # Load data
        data, wavelengths = self.load_hyperspectral_data(name)
        annotation = self.load_annotations(name)
        
        # Create RGB visualization
        rgb = self.get_rgb_visualization(data, wavelengths)
        
        # Get class statistics
        stats = self.get_class_statistics(annotation)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RGB image
        axes[0].imshow(rgb, interpolation='nearest')
        axes[0].set_title(f'RGB Visualization - {name}')
        axes[0].axis('off')
        
        # Annotation
        im = axes[1].imshow(annotation, interpolation='nearest', cmap='tab10')
        axes[1].set_title(f'Ground Truth Annotation - {name}')
        axes[1].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1])
        
        # Print statistics
        print(f"\nClass statistics for {name}:")
        for cls, stat in stats.items():
            print(f"Class {cls} ({stat['name']}): {stat['count']} pixels ({stat['percentage']:.2f}%)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()


def main():
    """
    Example usage of HyperBlood loader
    """
    # Initialize loader
    loader = HyperBloodLoader()
    
    # Test loading a single image
    print("Loading E_1 image...")
    data, wavelengths = loader.load_hyperspectral_data('E_1')
    annotation = loader.load_annotations('E_1')
    
    print(f"Data shape: {data.shape}")
    print(f"Wavelengths shape: {wavelengths.shape}")
    print(f"Annotation shape: {annotation.shape}")
    print(f"Spectral range: {wavelengths.min():.1f} - {wavelengths.max():.1f} nm")
    
    # Visualize the image
    loader.visualize_image('E_1')
    
    # Extract patches for training
    patches, labels = loader.create_patches(data, annotation)
    print(f"Extracted {len(patches)} patches")
    
    # Show class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution in patches:")
    for label, count in zip(unique_labels, counts):
        class_name = loader.class_names[label]
        print(f"Class {label} ({class_name}): {count} patches")


if __name__ == "__main__":
    main()
