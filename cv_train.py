import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from train import calculate_custom_accuracy, calculate_metrics, MaskedMSELoss, sigmoid
from cv_dataloader import TimeSeriesImageDataset, create_image_data_loader, TimeSeriesImageInferenceDataset
from cv_model import ViTTimeSeriesModel, ViTTimeSeriesModel_v2




class CVTrainer:
    """
    Trainer for CV-based time series model.
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        binary_loss_weight: float = 1.0,
        regression_loss_weight: float = 1.0,
        pos_weight: torch.Tensor = None,
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Loss functions
        self.binary_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.regression_criterion = MaskedMSELoss()
        
        # Loss weights for multi-task learning
        self.binary_loss_weight = binary_loss_weight
        self.regression_loss_weight = regression_loss_weight
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        total_binary_loss = 0
        total_regression_loss = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch in progress_bar:
            # Move batch to device
            images = batch['image'].to(self.device)
            binary_targets = batch['binary_targets'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            regression_masks = batch['regression_masks'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            binary_logits, regression_preds = self.model(images)
            
            # Calculate losses
            binary_loss = self.binary_criterion(binary_logits, binary_targets)
            regression_loss = self.regression_criterion(
                regression_preds, regression_targets, regression_masks
            )
            
            # Combine losses
            loss = self.binary_loss_weight * binary_loss + self.regression_loss_weight * regression_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            total_binary_loss += binary_loss.item()
            total_regression_loss += regression_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'binary_loss': binary_loss.item(),
                'regression_loss': regression_loss.item()
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(train_loader)
        avg_binary_loss = total_binary_loss / len(train_loader)
        avg_regression_loss = total_regression_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'binary_loss': avg_binary_loss,
            'regression_loss': avg_regression_loss
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        total_binary_loss = 0
        total_regression_loss = 0
        
        all_binary_logits = []
        all_binary_targets = []
        all_regression_preds = []
        all_regression_targets = []
        all_regression_masks = []
        all_file_names = []
        
        for batch in tqdm(val_loader, desc='Evaluating'):
            # Move batch to device
            images = batch['image'].to(self.device)
            binary_targets = batch['binary_targets'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            regression_masks = batch['regression_masks'].to(self.device)
            file_names = batch['file_name']
            
            # Forward pass
            binary_logits, regression_preds = self.model(images)
            
            # Calculate losses
            binary_loss = self.binary_criterion(binary_logits, binary_targets)
            regression_loss = self.regression_criterion(
                regression_preds, regression_targets, regression_masks
            )
            
            # Combine losses
            loss = self.binary_loss_weight * binary_loss + self.regression_loss_weight * regression_loss
            
            # Update metrics
            total_loss += loss.item()
            total_binary_loss += binary_loss.item()
            total_regression_loss += regression_loss.item()
            
            # Store predictions and targets for metric calculation
            all_binary_logits.append(binary_logits.cpu().numpy())
            all_binary_targets.append(binary_targets.cpu().numpy())
            all_regression_preds.append(regression_preds.cpu().numpy())
            all_regression_targets.append(regression_targets.cpu().numpy())
            all_regression_masks.append(regression_masks.cpu().numpy())
            all_file_names.extend(file_names)
        
        # Calculate average losses
        avg_loss = total_loss / len(val_loader)
        avg_binary_loss = total_binary_loss / len(val_loader)
        avg_regression_loss = total_regression_loss / len(val_loader)
        
        # Concatenate all predictions and targets
        binary_logits = np.concatenate(all_binary_logits)
        binary_preds = sigmoid(binary_logits)  # Apply sigmoid to get probabilities
        binary_targets = np.concatenate(all_binary_targets)
        regression_preds = np.concatenate(all_regression_preds)
        regression_targets = np.concatenate(all_regression_targets)
        regression_masks = np.concatenate(all_regression_masks)
        
        # Calculate additional metrics
        metrics = calculate_metrics(binary_preds, binary_targets, regression_preds, regression_targets, regression_masks)
        
        # Calculate custom accuracy
        custom_metrics = calculate_custom_accuracy(
            binary_preds=(binary_preds > 0.5).astype(int),  # Convert probabilities to binary predictions
            binary_targets=binary_targets,
            regression_preds=regression_preds,
            regression_targets=regression_targets,
            regression_masks=regression_masks,
            file_names=all_file_names,
            regression_threshold=0.1  # 10% threshold
        )
        
        return {
            'loss': avg_loss,
            'binary_loss': avg_binary_loss,
            'regression_loss': avg_regression_loss,
            'binary_preds': binary_preds,
            'binary_targets': binary_targets,
            'regression_preds': regression_preds,
            'regression_targets': regression_targets,
            'regression_masks': regression_masks,
            **metrics,
            **custom_metrics
        }




def train_cv_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    binary_loss_weight: float = 1.0,
    regression_loss_weight: float = 0.1,
    pos_weight: List[float] = None,
    warmup_epochs: int = 1,
) -> Tuple[nn.Module, Dict]:
    """
    Train the CV-based model and return the trained model and training history.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu')
        binary_loss_weight: Weight for binary classification loss
        regression_loss_weight: Weight for regression loss
        pos_weight: Positive weights for BCEWithLogitsLoss (list of weights for each binary target)
        warmup_epochs: Number of epochs for learning rate warmup
        
    Returns:
        Tuple of (trained model, training history)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Convert pos_weight to tensor if provided
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, device=device)
    
    # Initialize trainer
    trainer = CVTrainer(
        model=model,
        device=device,
        binary_loss_weight=binary_loss_weight,
        regression_loss_weight=regression_loss_weight,
        pos_weight=pos_weight
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_binary_loss': [],
        'train_regression_loss': [],
        'val_loss': [],
        'val_binary_loss': [],
        'val_regression_loss': [],
        'val_auroc': [],
        'val_mse': [],
        'val_mean_accuracy': []
    }
    
    # Training loop
    #best_val_loss = float('inf')
    best_val_mean_accuracy = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler)
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(val_loader)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_binary_loss'].append(train_metrics['binary_loss'])
        history['train_regression_loss'].append(train_metrics['regression_loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_binary_loss'].append(val_metrics['binary_loss'])
        history['val_regression_loss'].append(val_metrics['regression_loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_mse'].append(val_metrics['mse'])
        history['val_mean_accuracy'].append(val_metrics['mean_accuracy'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Binary Loss: {train_metrics['binary_loss']:.4f}, "
              f"Regression Loss: {train_metrics['regression_loss']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Binary Loss: {val_metrics['binary_loss']:.4f}, "
              f"Regression Loss: {val_metrics['regression_loss']:.4f}, "
              f"AUROC: {val_metrics['auroc']:.4f}, "
              f"MSE: {val_metrics['mse']:.4f}, "
              f"Mean Accuracy: {val_metrics['mean_accuracy']:.4f}")
        
        # Save best model
        if val_metrics['mean_accuracy'] > best_val_mean_accuracy:
            best_val_mean_accuracy = val_metrics['mean_accuracy']
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"New best model saved with validation mean accuracy: {best_val_mean_accuracy:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history


def inference_cv(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    binary_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Run inference on a dataset and return predictions as a pandas DataFrame.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing the data to predict
        device: Device to run inference on ('cuda' or 'cpu')
        binary_threshold: Threshold for converting binary probabilities to 0/1
    
    Returns:
        DataFrame with columns for file names, binary outputs, and regression outputs
    """
    model.eval()
    device = torch.device(device)
    model.to(device)
    
    all_binary_logits = []
    all_regression_preds = []
    all_file_names = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running inference"):
            # Move batch to device
            images = batch['image'].to(device)
            file_names = batch['file_name']
            
            # Forward pass
            binary_logits, regression_preds = model(images)
            
            # Store predictions
            all_binary_logits.append(binary_logits.cpu().numpy())
            all_regression_preds.append(regression_preds.cpu().numpy())
            all_file_names.extend(file_names)
    
    # Concatenate all predictions
    binary_logits = np.concatenate(all_binary_logits)
    binary_preds = sigmoid(binary_logits)  # Apply sigmoid to get probabilities
    regression_preds = np.concatenate(all_regression_preds)
    
    # Create DataFrame with results
    results = pd.DataFrame({'file_name': all_file_names})
    
    # Define column names
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
    
    # Add binary predictions (both probabilities and thresholded values)
    for i, col in enumerate(binary_cols):
        results[f'binary_{col}_prob'] = binary_preds[:, i]
        results[f'binary_{col}'] = (binary_preds[:, i] > binary_threshold).astype(int)
    
    # Add regression predictions
    for i, col in enumerate(regression_cols):
        results[f'regression_{col}'] = regression_preds[:, i]
    
    return results


def save_cv_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    output_path: str,
    device: str = 'cuda',
    binary_threshold: float = 0.5
) -> None:
    """
    Run inference and save predictions to a CSV file.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing the data to predict
        output_path: Path to save the CSV file
        device: Device to run inference on ('cuda' or 'cpu')
        binary_threshold: Threshold for converting binary probabilities to 0/1
    """
    results = inference_cv(model, dataloader, device, binary_threshold)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data loaders
    train_loader = create_image_data_loader(
        csv_path="path/to/train_labels.csv",
        images_dir="path/to/train_images",
        batch_size=32,
        shuffle=True
    )
    
    val_loader = create_image_data_loader(
        csv_path="path/to/val_labels.csv",
        images_dir="path/to/val_images",
        batch_size=32,
        shuffle=False
    )
    
    # Create model
    model = ViTTimeSeriesModel_v2(
        pretrained_model_name="google/vit-base-patch16-224",
        n_binary_targets=8,
        n_regression_targets=7,
        dropout=0.1,
        freeze_backbone=False,
        use_multi_layer_features=True
    )
    
    # Set positive weights for binary targets (if needed)
    pos_weights = [2.0, 3.0, 1.5, 2.5, 1.0, 3.5, 2.0, 1.5]  # Example weights
    
    # Train model
    trained_model, history = train_cv_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        device="cuda",
        binary_loss_weight=1.0,
        regression_loss_weight=0.1,
        pos_weight=pos_weights,
        warmup_epochs=2
    )
    
    # Save model
    torch.save(trained_model.state_dict(), "cv_model.pth")
    
    # Create inference dataset and loader
    inference_dataset = TimeSeriesImageInferenceDataset(
        images_dir="path/to/test_images"
    )
    
    inference_loader = DataLoader(
        inference_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run inference and save predictions
    save_cv_predictions(
        model=trained_model,
        dataloader=inference_loader,
        output_path="cv_predictions.csv",
        device="cuda"
    ) 