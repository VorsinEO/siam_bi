import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class TimeSeriesImageDataset(Dataset):
    """
    Dataset for loading time series data from PNG images.
    """
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        image_size: Tuple[int, int] = (224, 224),  # ViT standard input size
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to CSV file with labels
            images_dir: Directory containing PNG images of time series
            image_size: Size to resize images to (height, width)
            transform: Optional additional transformations
        """
        # Read labels
        if csv_path is not None:
            self.labels_df = pd.read_csv(csv_path)
        else:
            # If no labels provided, create empty DataFrame with file names
            image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
            self.labels_df = pd.DataFrame({'file_name': [os.path.splitext(f)[0] for f in image_files]})
            
            # Define target columns (same as in original dataset)
            binary_cols = [
                'Некачественное ГДИС', 'Влияние ствола скважины', 
                'Радиальный режим', 'Линейный режим', 'Билинейный режим', 
                'Сферический режим', 'Граница постоянного давления',
                'Граница непроницаемый разлом'
            ]
            
            regression_cols = [
                'Влияние ствола скважины_details', 'Радиальный режим_details',
                'Линейный режим_details', 'Билинейный режим_details',
                'Сферический режим_details', 'Граница постоянного давления_details',
                'Граница непроницаемый разлом_details'
            ]
            
            # Initialize with zeros
            self.labels_df[binary_cols] = 0
            self.labels_df[regression_cols] = 0
        
        # Store file names
        self.file_names = self.labels_df['file_name'].values
        
        # Store paths
        self.images_dir = images_dir
        
        # Define target columns
        self.binary_cols = [
            'Некачественное ГДИС', 'Влияние ствола скважины', 
            'Радиальный режим', 'Линейный режим', 'Билинейный режим', 
            'Сферический режим', 'Граница постоянного давления',
            'Граница непроницаемый разлом'
        ]
        
        self.regression_cols = [
            'Влияние ствола скважины_details', 'Радиальный режим_details',
            'Линейный режим_details', 'Билинейный режим_details',
            'Сферический режим_details', 'Граница постоянного давления_details',
            'Граница непроницаемый разлом_details'
        ]
        
        # Set up image transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get file name
        file_name = self.file_names[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{file_name}.png")
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image if loading fails
            image = torch.zeros((3, 224, 224))
        
        # Get labels
        row = self.labels_df[self.labels_df['file_name'] == file_name].iloc[0]
        binary_targets = row[self.binary_cols].values.astype(np.float32)
        regression_targets = row[self.regression_cols].values.astype(np.float32)
        
        # Create regression mask (True where regression target is not NaN)
        regression_mask = ~np.isnan(regression_targets)
        # Replace NaN with 0 for tensor conversion
        regression_targets = np.nan_to_num(regression_targets, nan=0.0)
        
        return {
            'image': image,  # [3, H, W]
            'binary_targets': torch.FloatTensor(binary_targets),
            'regression_targets': torch.FloatTensor(regression_targets),
            'regression_masks': torch.BoolTensor(regression_mask),
            'file_name': file_name
        }


def create_image_data_loader(
    csv_path: str,
    images_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (224, 224),
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
) -> DataLoader:
    """
    Create a DataLoader for time series images.
    
    Args:
        csv_path: Path to CSV file with labels
        images_dir: Directory containing PNG images
        batch_size: Batch size
        image_size: Size to resize images to
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        transform: Optional additional transformations
        
    Returns:
        DataLoader for the dataset
    """
    dataset = TimeSeriesImageDataset(
        csv_path=csv_path,
        images_dir=images_dir,
        image_size=image_size,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class TimeSeriesImageInferenceDataset(Dataset):
    """
    Dataset for inference on time series images without labels.
    """
    def __init__(
        self,
        images_dir: str,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            images_dir: Directory containing PNG images of time series
            image_size: Size to resize images to (height, width)
            transform: Optional additional transformations
        """
        # Get all PNG files in the directory
        self.file_names = [os.path.splitext(f)[0] for f in os.listdir(images_dir) if f.endswith('.png')]
        self.images_dir = images_dir
        
        # Set up image transformation
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get file name
        file_name = self.file_names[idx]
        
        # Load image
        img_path = os.path.join(self.images_dir, f"{file_name}.png")
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Create a blank image if loading fails
            image = torch.zeros((3, 224, 224))
        
        return {
            'image': image,  # [3, H, W]
            'file_name': file_name
        } 