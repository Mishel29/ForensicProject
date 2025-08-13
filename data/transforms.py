import cv2
import numpy as np
from typing import List, Tuple

def normalize_resolution(
    image: np.ndarray, 
    source_dpi: int, 
    target_dpi: int = 500
) -> np.ndarray:
    """Normalize image resolution to target DPI"""
    if source_dpi != target_dpi:
        scale_factor = target_dpi / source_dpi
        return cv2.resize(
            image, 
            None, 
            fx=scale_factor, 
            fy=scale_factor, 
            interpolation=cv2.INTER_AREA
        )
    return image

def enhance_contrast(image: np.ndarray, db_type: str) -> np.ndarray:
    """Apply sensor-specific contrast enhancement"""
    if db_type in ['DB1', 'DB2']:  # Optical sensors
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    else:  # Capacitive/synthetic
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(image)

def apply_sensor_specific_preprocessing(
    image: np.ndarray, 
    db_type: str
) -> np.ndarray:
    """Apply sensor-dependent preprocessing"""
    if db_type == 'DB1':  # Optical - reduce glare
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return cv2.addWeighted(image, 0.7, blurred, 0.3, 0)
    elif db_type == 'DB3':  # Capacitive - enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    return image

def generate_synthetic_latent(
    image: np.ndarray,
    db_type: str,
    rotation_range: List[float] = [-30, 30],
    noise_types: dict = None
) -> np.ndarray:
    """Generate realistic latent fingerprint from rolled print"""
    # Apply random rotation
    angle = np.random.uniform(*rotation_range)
    h, w = image.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
    
    # Apply sensor-specific noise
    if noise_types and db_type in noise_types:
        rotated = apply_noise(rotated, noise_types[db_type])
    
    # Extract random patch (40-80% of original)
    min_size = 0.4 if db_type != 'DB4' else 0.6
    crop_h = int(h * np.random.uniform(min_size, 0.8))
    crop_w = int(w * np.random.uniform(min_size, 0.8))
    x, y = np.random.randint(0, w - crop_w), np.random.randint(0, h - crop_h)
    patch = rotated[y:y+crop_h, x:x+crop_w]
    
    return cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA)

def apply_noise(image: np.ndarray, noise_types: List[str]) -> np.ndarray:
    """Apply multiple noise types to image"""
    noisy = image.copy()
    for noise_type in noise_types:
        if noise_type == "gaussian":
            noisy = noisy + np.random.normal(0, 15, noisy.shape)
        elif noise_type == "speckle":
            noisy = noisy + noisy * np.random.randn(*noisy.shape) * 0.1
        elif noise_type == "glare":
            glare = np.zeros_like(noisy)
            cv2.circle(glare, (noisy.shape[1]//2, noisy.shape[0]//2), 
                      noisy.shape[1]//3, 50, -1)
            noisy = cv2.addWeighted(noisy, 0.8, glare, 0.2, 0)
    return np.clip(noisy, 0, 255).astype(np.uint8)