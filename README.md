# Borehole Time Series Analysis

This project provides two complementary approaches for analyzing time series data from borehole measurements:

1. **Traditional Transformer Approach**: Processes raw time series data directly
2. **Computer Vision (CV) Approach**: Converts time series to images and applies vision models

## Approaches

### Traditional Transformer Approach
- Processes raw time series data with a custom transformer encoder
- Handles variable-length sequences with attention masking
- Supports both binary classification and regression tasks
- Files: `dataloader.py`, `model.py`, `train.py`

### CV-Based Approach
- Treats time series plots as images (1000x600 pixels)
- Supports multiple backbone architectures:
  - Vision Transformer (ViT)
  - EfficientNet
  - ConvNeXt
  - Swin Transformer
- Leverages transfer learning from pretrained models
- Files: `cv_dataloader.py`, `cv_model.py`, `cv_train.py`, `cv_example.py`

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Data Preparation and Analysis
The project includes several Jupyter notebooks for data analysis and preparation:

- `EDA.ipynb`: Data exploration and preprocessing
- `plot_them.ipynb`: Feature engineering, including:
  - Creating smoothed data
  - Computing derivatives
  - Generating time series plots for CV approach
- `train.ipynb`: Training process for the traditional approach
- `train_cv.ipynb`: Training process for the CV approach

### Traditional Approach
```python
from dataloader import create_data_loader
from model import TransformerModel
from train import train_model

# Create data loaders
train_loader = create_data_loader("train.csv", "data.parquet", batch_size=32)
val_loader = create_data_loader("val.csv", "data.parquet", batch_size=32)

# Create and train model
model = TransformerModel()
trained_model, history = train_model(model, train_loader, val_loader, num_epochs=20)
```

### CV Approach
```bash
# Train with different backbones
python cv_example.py --model_type vit --csv_path labels.csv --images_dir images
python cv_example.py --model_type efficientnet --model_name b3 --csv_path labels.csv --images_dir images
```

### Inference
The project includes an inference script for making predictions:
```bash
python infer.py --path_to_csv_dir "path/to/your/csv/directory"
```

## Key Features

- **Multi-task Learning**: Both approaches handle binary classification and regression simultaneously
- **Flexible Architecture**: Choose the best approach based on your data characteristics
- **Transfer Learning**: CV approach leverages pretrained models for better performance
- **Customizable**: Extensive options for hyperparameter tuning
- **Model Loading**: Easy loading of trained models with `from_pretrained` method
- **Comprehensive Analysis**: Full suite of Jupyter notebooks for data analysis and visualization

## When to Use Each Approach

- **Traditional Approach**: Better for raw time series with complex temporal dependencies
- **CV Approach**: Excels at capturing visual patterns and when domain experts analyze plots visually
- **Combined Approach**: For best results, use both approaches simultaneously to leverage their complementary strengths


## Requirements

- Python 3.8+
- PyTorch 1.10+
- See `requirements.txt` for full dependencies

## Project Structure

```
.
├── cv_dataloader.py     # Dataset classes for image-based time series
├── cv_model.py          # Model architectures with different backbones
├── cv_train.py          # Training and evaluation functions
├── cv_example.py        # Example script for training and inference
├── dataloader.py        # Data loading for traditional approach
├── model.py            # Transformer model implementation
├── train.py            # Training utilities
├── infer.py            # Inference script
├── EDA.ipynb           # Data exploration notebook
├── plot_them.ipynb     # Feature engineering and plotting
├── train.ipynb         # Training notebook
├── train_cv.ipynb      # CV training notebook
├── requirements.txt     # Project dependencies
└── README.md            # This file
```

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@software{timeseries_image_analysis,
  author = {Evgeny Vorsin},
  title = {Borehole Time Series Analysis},
  year = {2025},
  url = {https://github.com/VorsinEO/siam_bi}
}
```