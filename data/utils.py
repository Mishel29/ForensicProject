import os
import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
from typing import Tuple

def generate_ground_truth(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate skeleton, orientation field, and minutiae map"""
    # Binarization
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Skeletonization
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    
    # Orientation field
    orientation = compute_orientation_field(binary)
    
    # Minutiae detection
    minutiae = detect_minutiae(skeleton)
    
    return skeleton, orientation, minutiae

def compute_orientation_field(binary_img: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Compute fingerprint orientation field"""
    h, w = binary_img.shape
    orientation = np.zeros((h//block_size, w//block_size))
    
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            block = binary_img[i:i+block_size, j:j+block_size]
            gx = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
            Gxx, Gyy, Gxy = np.sum(gx**2), np.sum(gy**2), np.sum(gx*gy)
            theta = 0.5 * np.arctan2(2*Gxy, Gxx-Gyy) + np.pi/2
            orientation[i//block_size, j//block_size] = theta
    
    return cv2.resize(orientation, (w, h), interpolation=cv2.INTER_NEAREST)

def detect_minutiae(skeleton: np.ndarray, r: int = 5) -> np.ndarray:
    """Detect minutiae points from skeleton"""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    padded = cv2.copyMakeBorder(skeleton, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    minutiae_map = np.zeros_like(skeleton)
    
    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            if skeleton[y,x] == 0:
                continue
            patch = padded[y:y+3, x:x+3]
            crossings = np.sum(kernel * patch) // 255
            if crossings in [1, 3]:  # Endpoints or bifurcations
                minutiae_map[y,x] = 255
    
    # Non-max suppression
    for y in range(r, minutiae_map.shape[0]-r):
        for x in range(r, minutiae_map.shape[1]-r):
            if minutiae_map[y,x] == 255:
                neighborhood = minutiae_map[y-r:y+r+1, x-r:x+r+1]
                if np.sum(neighborhood) > 255:
                    minutiae_map[y-r:y+r+1, x-r:x+r+1] = 0
                    minutiae_map[y,x] = 255
                    
    return minutiae_map

def save_processed_data(
    output_dir: Path,
    db: str,
    base_name: str,
    latent: np.ndarray,
    skeleton: np.ndarray,
    orientation: np.ndarray,
    minutiae: np.ndarray
) -> None:
    """Save all processed data components"""
    cv2.imwrite(str(output_dir / 'latent' / f'{db}_{base_name}.png'), latent)
    cv2.imwrite(str(output_dir / 'skeleton' / f'{db}_{base_name}.png'), skeleton)
    np.save(str(output_dir / 'orientation' / f'{db}_{base_name}.npy'), orientation)
    cv2.imwrite(str(output_dir / 'minutiae' / f'{db}_{base_name}.png'), minutiae)

def validate_output(output_dir: Path, db: str) -> bool:
    """Validate that processing completed successfully"""
    latent_files = list((output_dir / 'latent').glob(f'{db}_*.png'))
    skeleton_files = list((output_dir / 'skeleton').glob(f'{db}_*.png'))
    
    if len(latent_files) == 0 or len(latent_files) != len(skeleton_files):
        print(f"Validation failed for {db}: File counts don't match")
        return False
    
    print(f"Successfully processed {len(latent_files)} images from {db}")
    return True