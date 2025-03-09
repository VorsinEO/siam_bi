import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve
import copy

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores entries where mask is False"""
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # mask: True where we should calculate loss, False where we should ignore
        masked_diff = (pred - target) * mask.float()
        squared_diff = masked_diff ** 2
        
        if self.reduction == 'mean':
            # Normalize by the number of unmasked elements
            num_unmasked = mask.sum().item()
            loss = squared_diff.sum() / (num_unmasked + 1e-8)
        else:  # sum
            loss = squared_diff.sum()
            
        return loss

class Trainer:
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
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            binary_targets = batch['binary_targets'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            regression_masks = batch['regression_masks'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            binary_logits, regression_preds = self.model(input_features, attention_mask)
            
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
            input_features = batch['input_features'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            binary_targets = batch['binary_targets'].to(self.device)
            regression_targets = batch['regression_targets'].to(self.device)
            regression_masks = batch['regression_masks'].to(self.device)
            file_names = batch['file_name']
            
            # Forward pass
            binary_logits, regression_preds = self.model(input_features, attention_mask)
            
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

def sigmoid(x):
    """Apply sigmoid function to numpy array"""
    return 1 / (1 + np.exp(-x))

def calculate_metrics(binary_preds, binary_targets, regression_preds, regression_targets, regression_masks):
    # Binary classification metrics
    auroc = roc_auc_score(binary_targets, binary_preds, average='macro')
    
    # Regression metrics (only for unmasked values)
    mse = np.mean(((regression_preds - regression_targets) * regression_masks) ** 2)
    
    return {
        'auroc': auroc,
        'mse': mse
    }

def calculate_custom_accuracy(
    binary_preds: np.ndarray,
    binary_targets: np.ndarray,
    regression_preds: np.ndarray,
    regression_targets: np.ndarray,
    regression_masks: np.ndarray,
    file_names: List[str],
    regression_threshold: float = 0.1  # 10% threshold
) -> Dict[str, float]:
    """
    Calculate custom accuracy metric:
    1. For binary targets: 1 point for each TP or TN
    2. For regression targets: 1 point if abs difference <= 10% (for non-NaN targets)
    3. Calculate share of max possible points for each file
    4. Return mean accuracy across all files
    
    Args:
        binary_preds: Binary predictions (n_samples, n_binary_targets)
        binary_targets: Binary targets (n_samples, n_binary_targets)
        regression_preds: Regression predictions (n_samples, n_regression_targets)
        regression_targets: Regression targets (n_samples, n_regression_targets)
        regression_masks: Boolean masks for regression targets (n_samples, n_regression_targets)
        file_names: List of file names corresponding to predictions
        regression_threshold: Threshold for regression accuracy (default 0.1 = 10%)
    
    Returns:
        Dictionary containing:
        - mean_accuracy: Mean accuracy across all files
        - file_accuracies: List of accuracies for each file
        - file_names: List of file names
    """
    # Convert file_names to numpy array if it's not already
    file_names = np.array(file_names)
    
    # Get unique file names
    unique_files = np.unique(file_names)
    file_accuracies = []
    split_accuracies = []
    
    for file_name in unique_files:
        # Get indices for this file using boolean indexing
        file_indices = file_names == file_name
        
        # Calculate binary accuracy
        file_binary_preds = binary_preds[file_indices]
        file_binary_targets = binary_targets[file_indices]
        #print(file_binary_preds.shape, file_binary_targets.shape)
        binary_correct = (file_binary_preds == file_binary_targets).sum()
        max_binary_points = file_binary_preds.size
        mismatch_indices = find_mismatch_indices(file_binary_preds, file_binary_targets)
        # Calculate regression accuracy
        file_regression_preds = regression_preds[file_indices]
        file_regression_targets = regression_targets[file_indices]
        file_regression_masks = regression_masks[file_indices]
        if len(mismatch_indices) > 0:
            file_regression_preds[0,mismatch_indices]= 1000000
        #print(file_regression_preds.shape, file_regression_targets.shape, file_regression_masks.shape)
        # Calculate relative differences for non-NaN targets
        valid_regression = file_regression_masks
        if valid_regression.any():
            rel_diff = np.abs(file_regression_preds - file_regression_targets) / (file_regression_targets + 1e-8)
            rel_diff = np.abs(rel_diff)
            regression_correct = ((rel_diff <= regression_threshold) & valid_regression).sum()
            max_regression_points = valid_regression.sum()
        else:
            regression_correct = 0
            max_regression_points = 0
        
        # Calculate total accuracy for this file
        #total_correct = binary_correct + regression_correct
        #max_points = max_binary_points + max_regression_points
        
        if max_binary_points > 0:
            bin_accuracy = binary_correct / max_binary_points
        else:
            bin_accuracy = 0.0
        if max_regression_points > 0:
            reg_accuracy = regression_correct / max_regression_points
        else:
            reg_accuracy = 0.0
        #print(bin_accuracy, reg_accuracy)
        file_accuracy = 0.7*bin_accuracy + 0.3* reg_accuracy
        file_accuracies.append(file_accuracy)
        split_accuracies.append((bin_accuracy, reg_accuracy))

    # Calculate mean accuracy
    mean_accuracy = np.mean(file_accuracies)
    
    return {
        'mean_accuracy': mean_accuracy,
        'file_accuracies': file_accuracies,
        'split_accuracies': split_accuracies,
        #'file_names': unique_files.tolist()
    }

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float = 1e-4,
    device: str = 'cuda',
    binary_loss_weight: float = 1.0,
    regression_loss_weight: float = 0.1,
    pos_weight: List[float] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Train the model and return the trained model and training history.
    
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
        
    Returns:
        Tuple of (trained model, training history)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Convert pos_weight to tensor if provided
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight, device=device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=device,
        binary_loss_weight=binary_loss_weight,
        regression_loss_weight=regression_loss_weight,
        pos_weight=pos_weight
    )
    
    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # 10% warmup
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
    best_val_loss = float('inf')
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

def inference(
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
            input_features = batch['input_features'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            file_names = batch['file_name']
            
            # Forward pass
            binary_logits, regression_preds = model(input_features, attention_mask)
            
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

def save_predictions(
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
    results = inference(model, dataloader, device, binary_threshold)
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}") 

def find_mismatch_indices(a, b):
    """
    Find indices where b has 1 but a has 0, excluding the first element.

    Parameters:
        a (numpy.ndarray): First binary array.
        b (numpy.ndarray): Second binary array.

    Returns:
        list: List of indices where b[i] == 1 and a[i] == 0, excluding the first element.
    """
    # Ensure both arrays are numpy arrays and have the same shape
    a = np.array(a).flatten()[1:]  # Flatten and exclude the first element
    b = np.array(b).flatten()[1:]  # Flatten and exclude the first element
    
    # Find indices where b is 1 and a is 0
    mismatch_indices = np.where((b == 1) & (a == 0))[0]
    
    return mismatch_indices.tolist()