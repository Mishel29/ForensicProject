"""
Blood Detection Demo Script
===========================

Simple demo script to test the HyperBlood dataset loading and basic functionality
without requiring all the heavy dependencies.

This script demonstrates:
1. Dataset structure exploration
2. Basic data loading (using alternative methods if spectral library not available)
3. Data visualization using matplotlib
4. Class distribution analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def explore_dataset_structure(dataset_path: str):
    """
    Explore the HyperBlood dataset structure
    
    Args:
        dataset_path: Path to HyperBlood dataset
    """
    print("="*50)
    print("HYPERBLOOD DATASET EXPLORATION")
    print("="*50)
    
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please check the dataset path.")
        return False
    
    print(f"✅ Dataset found at: {dataset_path}")
    
    # Check main directories
    subdirs = ['data', 'anno', 'images', 'src']
    for subdir in subdirs:
        subdir_path = dataset_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob('*'))
            print(f"📁 {subdir}/: {len(files)} files")
            if len(files) <= 10:  # Show files if not too many
                for file in files[:5]:
                    print(f"   - {file.name}")
                if len(files) > 5:
                    print(f"   ... and {len(files) - 5} more files")
        else:
            print(f"❌ {subdir}/ not found")
    
    # List available hyperspectral images
    data_dir = dataset_dir / 'data'
    if data_dir.exists():
        hdr_files = list(data_dir.glob('*.hdr'))
        print(f"\n📊 Available hyperspectral images: {len(hdr_files)}")
        for hdr_file in sorted(hdr_files):
            base_name = hdr_file.stem
            float_file = data_dir / f"{base_name}.float"
            if float_file.exists():
                print(f"   ✅ {base_name}")
            else:
                print(f"   ❌ {base_name} (missing .float file)")
    
    return True

def load_annotation_simple(anno_path: str):
    """
    Load annotation using numpy (fallback method)
    
    Args:
        anno_path: Path to annotation .npz file
        
    Returns:
        Annotation array or None if failed
    """
    try:
        data = np.load(anno_path)
        if 'gt' in data:
            return data['gt']
        else:
            print(f"Available keys in {anno_path}: {list(data.keys())}")
            return None
    except Exception as e:
        print(f"❌ Failed to load annotation: {e}")
        return None

def analyze_image_info(dataset_path: str, image_name: str):
    """
    Analyze information about a specific image
    
    Args:
        dataset_path: Path to HyperBlood dataset
        image_name: Name of image to analyze (e.g., 'E_1')
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING IMAGE: {image_name}")
    print(f"{'='*50}")
    
    dataset_dir = Path(dataset_path)
    
    # Check if files exist
    hdr_file = dataset_dir / 'data' / f'{image_name}.hdr'
    float_file = dataset_dir / 'data' / f'{image_name}.float'
    anno_file = dataset_dir / 'anno' / f'{image_name}.npz'
    
    print(f"📄 Header file: {'✅' if hdr_file.exists() else '❌'} {hdr_file}")
    print(f"📄 Data file: {'✅' if float_file.exists() else '❌'} {float_file}")
    print(f"📄 Annotation file: {'✅' if anno_file.exists() else '❌'} {anno_file}")
    
    # Try to read header file
    if hdr_file.exists():
        try:
            with open(hdr_file, 'r') as f:
                header_content = f.read()
            
            # Extract basic info from header
            lines = header_content.split('\n')
            info = {}
            for line in lines:
                if '=' in line:
                    key, value = line.split('=', 1)
                    info[key.strip()] = value.strip()
            
            print(f"\n📊 Image Information:")
            if 'samples' in info:
                print(f"   Width: {info['samples']} pixels")
            if 'lines' in info:
                print(f"   Height: {info['lines']} pixels")
            if 'bands' in info:
                print(f"   Bands: {info['bands']} spectral channels")
            if 'data type' in info:
                print(f"   Data type: {info['data type']}")
            
            # Try to get wavelength info
            wavelength_info = [line for line in lines if 'wavelength' in line.lower()]
            if wavelength_info:
                print(f"   Spectral info available: ✅")
            
        except Exception as e:
            print(f"❌ Failed to read header: {e}")
    
    # Analyze annotation
    if anno_file.exists():
        annotation = load_annotation_simple(str(anno_file))
        if annotation is not None:
            print(f"\n🏷️ Annotation Analysis:")
            print(f"   Shape: {annotation.shape}")
            print(f"   Data type: {annotation.dtype}")
            
            # Class distribution
            unique_classes, counts = np.unique(annotation, return_counts=True)
            total_pixels = annotation.size
            
            class_names = {
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
            
            print(f"   Total pixels: {total_pixels:,}")
            print(f"   Classes found: {len(unique_classes)}")
            print(f"\n   Class distribution:")
            for cls, count in zip(unique_classes, counts):
                class_name = class_names.get(cls, f'unknown_{cls}')
                percentage = (count / total_pixels) * 100
                print(f"     {cls:2d} ({class_name:15s}): {count:8,} pixels ({percentage:5.2f}%)")
    
    return hdr_file.exists() and float_file.exists() and anno_file.exists()

def create_simple_visualization(dataset_path: str, image_name: str):
    """
    Create simple visualization of annotation data
    
    Args:
        dataset_path: Path to HyperBlood dataset
        image_name: Name of image to visualize
    """
    print(f"\n📈 Creating visualization for {image_name}...")
    
    dataset_dir = Path(dataset_path)
    anno_file = dataset_dir / 'anno' / f'{image_name}.npz'
    
    if not anno_file.exists():
        print(f"❌ Annotation file not found: {anno_file}")
        return
    
    # Load annotation
    annotation = load_annotation_simple(str(anno_file))
    if annotation is None:
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Annotation map
    im1 = axes[0].imshow(annotation, cmap='tab10', interpolation='nearest')
    axes[0].set_title(f'Class Annotation - {image_name}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Class distribution
    unique_classes, counts = np.unique(annotation, return_counts=True)
    class_names = {
        0: 'background', 1: 'blood', 2: 'ketchup', 3: 'artificial_blood',
        4: 'beetroot_juice', 5: 'poster_paint', 6: 'tomato_concentrate', 
        7: 'acrylic_paint', 8: 'uncertain_blood'
    }
    
    labels = [f"{cls}: {class_names.get(cls, 'unknown')}" for cls in unique_classes]
    colors = plt.cm.tab10(unique_classes / 10.0)
    
    axes[1].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%')
    axes[1].set_title(f'Class Distribution - {image_name}')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('./demo_results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{image_name}_visualization.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_path}")
    
    plt.show()

def test_dataset_availability():
    """
    Test different possible dataset locations
    """
    possible_paths = [
        "../data/raw/HyperBlood/",
        "./data/raw/HyperBlood/",
        "../../data/raw/HyperBlood/",
        "./HyperBlood/",
        "../HyperBlood/"
    ]
    
    print("🔍 Searching for HyperBlood dataset...")
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"✅ Found dataset at: {path}")
            return str(Path(path).resolve())
    
    print("❌ HyperBlood dataset not found in common locations:")
    for path in possible_paths:
        print(f"   {path}")
    
    print("\n💡 Please ensure the HyperBlood dataset is available and update the path.")
    return None

def main():
    """
    Main demo function
    """
    print("🩸 BLOOD DETECTION DEMO")
    print("=" * 50)
    
    # Find dataset
    dataset_path = test_dataset_availability()
    if dataset_path is None:
        print("\n❌ Cannot proceed without dataset")
        return
    
    # Explore dataset structure
    if not explore_dataset_structure(dataset_path):
        return
    
    # Test specific images
    test_images = ['E_1', 'F_1', 'E_21', 'F_1a']
    
    print(f"\n🧪 Testing {len(test_images)} sample images...")
    
    available_images = []
    for image_name in test_images:
        print(f"\n{'-' * 30}")
        if analyze_image_info(dataset_path, image_name):
            available_images.append(image_name)
            print(f"✅ {image_name} is ready for processing")
        else:
            print(f"❌ {image_name} has missing files")
    
    if available_images:
        print(f"\n📊 Creating visualizations for available images...")
        for image_name in available_images[:2]:  # Limit to first 2 to avoid too many plots
            try:
                create_simple_visualization(dataset_path, image_name)
            except Exception as e:
                print(f"❌ Failed to visualize {image_name}: {e}")
    
    # Summary
    print(f"\n{'=' * 50}")
    print("DEMO SUMMARY")
    print(f"{'=' * 50}")
    print(f"✅ Dataset location: {dataset_path}")
    print(f"✅ Available images: {len(available_images)}/{len(test_images)}")
    print(f"✅ Ready for blood detection training: {'Yes' if available_images else 'No'}")
    
    if available_images:
        print(f"\n🚀 Next steps:")
        print(f"1. Install required packages: pip install -r requirements.txt")
        print(f"2. Run training: python train_blood_detection.py")
        print(f"3. Run evaluation: python evaluate_blood_detection.py")
    else:
        print(f"\n⚠️ Please ensure all required dataset files are available")

if __name__ == "__main__":
    main()
