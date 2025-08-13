# Blood Detection using Hyperspectral Imaging

This project implements the **Fast Extraction (FE) framework** for forensic blood detection using hyperspectral imaging, based on the research paper:

> **"Enhancing forensic blood detection using hyperspectral imaging and advanced preprocessing techniques"**  
> *by Dalal AL-Alimi and Mohammed A.A. Al-qaness*

## 🩸 Overview

The system uses the HyperBlood dataset to classify different substances that might be confused with blood in forensic investigations:

- **Blood** (human blood)
- **Ketchup** 
- **Artificial blood**
- **Beetroot juice**
- **Poster paint**
- **Tomato concentrate** 
- **Acrylic paint**

## 🏗️ Architecture

### Fast Extraction (FE) Framework

The FE framework consists of two main stages:

1. **Preprocessing Stage**: Enhancing Transformation Reduction (ETR)
   - Dimension reduction using enhanced covariance matrix
   - Morphological dilation for pixel position correction
   - Gaussian distribution normalization

2. **Classification Stage**: Convolutional Neural Network
   - Two Conv2D layers with ELU activation
   - Dropout for regularization  
   - Dense layers for final classification

## 📁 Project Structure

```
blood_detection/
├── configs/
│   └── config.yaml              # Configuration parameters
├── src/
│   └── etr_preprocessing.py     # ETR preprocessing implementation
├── models/
│   └── fe_classifier.py         # FE classification model
├── utils/
│   └── hyperblood_loader.py     # HyperBlood dataset loader
├── results/                     # Training and evaluation results
├── demo.py                      # Quick demo script
├── train_blood_detection.py     # Main training script
├── evaluate_blood_detection.py  # Comprehensive evaluation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or navigate to the project directory
cd blood_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Preparation

Ensure you have the HyperBlood dataset available. The dataset should be structured as:

```
data/raw/HyperBlood/
├── data/          # Hyperspectral images (.hdr and .float files)
├── anno/          # Ground truth annotations (.npz files)
├── images/        # RGB visualizations
└── src/           # Original dataset API
```

### 3. Run Demo

Test the dataset and basic functionality:

```bash
python demo.py
```

This will:
- Explore the dataset structure
- Analyze sample images
- Create basic visualizations
- Verify everything is working

### 4. Train the Model

Run the complete training pipeline:

```bash
python train_blood_detection.py --preprocessing etr --n_components 50
```

Options:
- `--preprocessing`: Choose `etr` or `pca`
- `--n_components`: Number of components for dimensionality reduction
- `--patch_size`: Size of spatial patches (default: 9)
- `--results_dir`: Directory to save results

### 5. Evaluate Performance

Run comprehensive evaluation:

```bash
python evaluate_blood_detection.py
```

This generates:
- Performance comparison plots
- Detailed evaluation report
- Method ranking and analysis

## 📊 Expected Results

Based on the paper, the FE framework with ETR preprocessing achieves:

- **97-100% accuracy** across different HyperBlood images
- **Superior performance** compared to baseline deep learning models
- **Robust performance** across different blood aging periods
- **Efficient processing** with reduced computational time

### Performance by Dataset

| Dataset | Complexity | Expected Accuracy |
|---------|------------|-------------------|
| E_1     | High       | ~99%             |
| E_21    | Very High  | ~97%             |
| F_1     | Medium     | ~100%            |
| F_1a    | Medium     | ~100%            |

## 🔧 Configuration

Modify `configs/config.yaml` to customize:

- **Dataset paths** and train/test splits
- **Preprocessing parameters** (ETR vs PCA, components)
- **Model architecture** (filters, layers, activation)
- **Training parameters** (epochs, batch size, learning rate)
- **Evaluation metrics** and visualization options

## 📚 Key Components

### ETR Preprocessing (`src/etr_preprocessing.py`)

Implements the Enhancing Transformation Reduction method:

```python
from src.etr_preprocessing import ETRPreprocessor

# Initialize ETR
etr = ETRPreprocessor(n_components=50, enhancement_factor=0.1)

# Fit and transform data
processed_data = etr.fit_transform(hyperspectral_data)
```

### FE Classifier (`models/fe_classifier.py`)

The Fast Extraction classification model:

```python
from models.fe_classifier import FEClassifier

# Initialize classifier
classifier = FEClassifier(input_shape=(9, 9, 50), num_classes=8)

# Train model
classifier.train(X_train, y_train, X_val, y_val)
```

### Dataset Loader (`utils/hyperblood_loader.py`)

Utilities for loading and processing the HyperBlood dataset:

```python
from utils.hyperblood_loader import HyperBloodLoader

# Initialize loader
loader = HyperBloodLoader(dataset_path)

# Load hyperspectral data
data, wavelengths = loader.load_hyperspectral_data('E_1')
annotation = loader.load_annotations('E_1')
```

## 🧪 Experimental Setup

### Training Images
- **E_1**: Scene E, Day 1 (4h 40m aging)
- **E_21**: Scene E, Day 21 (21 days aging)

### Test Images  
- **F_1**: Scene F, Day 1 (1h 20m aging)
- **F_1a**: Scene F, Day 1 (5h 50m aging)

### Evaluation Metrics
- **Overall Accuracy (OA)**
- **Average Accuracy (AA)** 
- **Kappa Accuracy (KA)**
- **Per-class Recall**

## 🔬 Method Comparison

The evaluation compares multiple approaches:

1. **FE + ETR** (Proposed method)
2. **FE + PCA** (Alternative preprocessing)
3. **1D-CNN + PCA** (Spectral-only baseline)
4. **2D-CNN + PCA** (Spatial-spectral baseline)
5. **MLP + PCA** (Simple baseline)

## 📈 Visualization

The system generates comprehensive visualizations:

- **Training curves** (accuracy and loss)
- **Confusion matrices** for each test set
- **Performance comparison** across methods
- **Class distribution** analysis
- **RGB visualizations** of hyperspectral data

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages in `requirements.txt` are installed
2. **Dataset Not Found**: Check the dataset path in config or use `--dataset_path`
3. **Memory Issues**: Reduce batch size or patch size in configuration
4. **GPU Issues**: Set `use_gpu: false` in config for CPU-only training

### Dependencies

Key packages required:
- `tensorflow >= 2.8.0` (or `torch >= 1.10.0`)
- `scikit-learn >= 1.0.0`
- `spectral >= 0.22.0` (for hyperspectral image handling)
- `numpy`, `matplotlib`, `seaborn`

## 📄 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{al2025enhancing,
  title={Enhancing forensic blood detection using hyperspectral imaging and advanced preprocessing techniques},
  author={AL-Alimi, Dalal and Al-qaness, Mohammed A.A.},
  journal={Talanta},
  volume={283},
  pages={127097},
  year={2025},
  publisher={Elsevier}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest improvements
- Add new features
- Improve documentation

## 📝 License

This project is for educational and research purposes. Please refer to the original paper and HyperBlood dataset licensing terms.

## 🔗 Related Work

- **HyperBlood Dataset**: [DOI: 10.5281/zenodo.3984905](https://doi.org/10.5281/zenodo.3984905)
- **Original Paper**: [Talanta 283 (2025) 127097](https://doi.org/10.1016/j.talanta.2024.127097)
- **Hyperspectral Processing**: [Spectral Python Library](https://github.com/spectralpython/spectral)

