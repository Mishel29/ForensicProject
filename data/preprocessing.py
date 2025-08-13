import yaml
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .transforms import (
    normalize_resolution,
    enhance_contrast,
    apply_sensor_specific_preprocessing,
    generate_synthetic_latent
)
from .utils import (
    generate_ground_truth,
    save_processed_data,
    validate_output
)

def load_config(config_path="configs/preprocessing.yaml"):
    """Load preprocessing configuration"""
    with open(config_path) as f:
        return yaml.safe_load(f)

def process_database(config):
    """Main preprocessing pipeline"""
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    
    # Create output directories
    (output_dir / 'latent').mkdir(parents=True, exist_ok=True)
    (output_dir / 'skeleton').mkdir(parents=True, exist_ok=True)
    (output_dir / 'orientation').mkdir(parents=True, exist_ok=True)
    (output_dir / 'minutiae').mkdir(parents=True, exist_ok=True)

    for db in config['databases']:
        db_dir = input_dir / db
        print(f"\nProcessing {db} database...")
        
        for img_path in tqdm(list(db_dir.glob('*.tif')), desc=db):
            try:
                # Load and validate image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError(f"Failed to load {img_path}")

                # Apply preprocessing pipeline
                img = normalize_resolution(
                    img, 
                    config['resolutions'][db], 
                    config['target_resolution']
                )
                img = enhance_contrast(img, db)
                img = apply_sensor_specific_preprocessing(img, db)
                
                # Generate synthetic latent and ground truth
                latent = generate_synthetic_latent(
                    img, 
                    db,
                    rotation_range=config['augmentation']['rotation_range'],
                    noise_types=config['augmentation']['noise_types']
                )
                skeleton, orientation, minutiae = generate_ground_truth(img)
                
                # Save processed data
                save_processed_data(
                    output_dir,
                    db,
                    img_path.stem,
                    latent,
                    skeleton,
                    orientation,
                    minutiae
                )

            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
                continue

        # Validate output for this database
        validate_output(output_dir, db)

if __name__ == "__main__":
    config = load_config()
    process_database(config)