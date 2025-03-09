import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm
class WellDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        parquet_path: str,
        max_seq_len: int = 512,
    ):

        
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
        
        # Read time series data and preprocess
        self.timeseries_df = pd.read_parquet(parquet_path)
        self.timeseries_cols = self.timeseries_df.columns[1:]
        # Read the labels data
        if csv_path is not None:
            self.labels_df = pd.read_csv(csv_path)
        else:
            self.labels_df = pd.DataFrame({'file_name': self.timeseries_df['file_name'].unique()})
            self.labels_df[self.binary_cols] = 0
            self.labels_df[self.regression_cols] = 0
        self.max_seq_len = max_seq_len
        
        # Get unique file names (well IDs)
        self.file_names = self.labels_df['file_name'].unique()
        
        # Preprocess labels for faster access
        self.label_dict = {}
        for _, row in self.labels_df.iterrows():
            file_name = row['file_name']
            
            # Convert binary targets to float32 numpy array
            binary_targets = row[self.binary_cols].values.astype(np.float32)
            
            # Handle regression targets - explicitly convert to float32 and handle NaNs
            regression_values = row[self.regression_cols].values
            regression_mask = ~pd.isna(regression_values)
            regression_targets = np.zeros(len(self.regression_cols), dtype=np.float32)
            
            for i, val in enumerate(regression_values):
                if not pd.isna(val):
                    regression_targets[i] = float(val)
            
            self.label_dict[file_name] = {
                'binary': binary_targets,
                'regression': regression_targets,
                'regression_masks': regression_mask
            }
        
        # Preprocess and group time series data by file_name
        print("Preprocessing time series data...")
        self.timeseries_dict = {}
        
        try:
            from tqdm import tqdm
        except ImportError:
            # Define a simple tqdm replacement if tqdm is not installed
            def tqdm(iterable, *args, **kwargs):
                return iterable
        
        for file_name, group in tqdm(self.timeseries_df.groupby('file_name')):
            # Only preprocess files that are in our file_names list
            if file_name in self.file_names:
                # Ensure time series data is float32
                time_series = group[self.timeseries_cols].values.astype(np.float32)
                
                # Pad or truncate sequence
                if len(time_series) > self.max_seq_len:
                    time_series = time_series[:self.max_seq_len]
                    attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
                else:
                    pad_length = self.max_seq_len - len(time_series)
                    time_series = np.pad(
                        time_series,
                        ((0, pad_length), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                    attention_mask = np.ones(self.max_seq_len, dtype=np.float32)
                    attention_mask[len(group):] = 0
                
                self.timeseries_dict[file_name] = {
                    'time_series': time_series,
                    'attention_mask': attention_mask
                }
        
        print(f"Preprocessed {len(self.timeseries_dict)} time series")
        
        # Free up memory
        del self.timeseries_df
        import gc
        gc.collect()

    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_name = self.file_names[idx]
        
        # Get preprocessed time series data
        time_series_data = self.timeseries_dict.get(file_name)
        if time_series_data is None:
            # Handle missing data (should not happen if preprocessing was correct)
            print(f"Warning: Missing time series for {file_name}")
            time_series = np.zeros((self.max_seq_len, 3), dtype=np.float32)
            attention_mask = np.zeros(self.max_seq_len, dtype=np.float32)
        else:
            time_series = time_series_data['time_series']
            attention_mask = time_series_data['attention_mask']
        
        # Get preprocessed labels
        label_data = self.label_dict[file_name]
        binary_targets = label_data['binary']
        regression_targets = label_data['regression']
        regression_mask = label_data['regression_masks']
            
        return {
            'input_features': torch.FloatTensor(time_series),
            'attention_mask': torch.FloatTensor(attention_mask),
            'binary_targets': torch.FloatTensor(binary_targets),
            'regression_targets': torch.FloatTensor(regression_targets),
            'regression_masks': torch.BoolTensor(regression_mask),
            'file_name': file_name
        }

def create_data_loader(
    csv_path: str,
    parquet_path: str,
    batch_size: int = 32,
    max_seq_len: int = 512,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for the well data.
    """
    dataset = WellDataset(
        csv_path=csv_path,
        parquet_path=parquet_path,
        max_seq_len=max_seq_len
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
