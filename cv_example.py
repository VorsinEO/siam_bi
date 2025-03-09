import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from cv_dataloader import TimeSeriesImageDataset, create_image_data_loader
from cv_model import create_model, TimeSeriesImageModel
from cv_train import train_cv_model, save_cv_predictions


def plot_training_history(history, output_path='training_history.png'):
    """Plot training history metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot binary losses
    plt.subplot(2, 2, 2)
    plt.plot(history['train_binary_loss'], label='Train Binary Loss')
    plt.plot(history['val_binary_loss'], label='Val Binary Loss')
    plt.title('Binary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot regression losses
    plt.subplot(2, 2, 3)
    plt.plot(history['train_regression_loss'], label='Train Regression Loss')
    plt.plot(history['val_regression_loss'], label='Val Regression Loss')
    plt.title('Regression Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(2, 2, 4)
    plt.plot(history['val_auroc'], label='AUROC')
    plt.plot(history['val_mean_accuracy'], label='Mean Accuracy')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_model_info(model_type):
    """Get model-specific information."""
    if model_type == 'vit':
        return {
            'name': 'Vision Transformer',
            'default_model': 'google/vit-base-patch16-224',
            'learning_rate': 2e-5,
            'batch_size': 16
        }
    elif model_type == 'efficientnet':
        return {
            'name': 'EfficientNet',
            'default_model': 'b0',
            'learning_rate': 1e-4,
            'batch_size': 32
        }
    elif model_type == 'convnext':
        return {
            'name': 'ConvNeXt',
            'default_model': 'facebook/convnext-tiny-224',
            'learning_rate': 5e-5,
            'batch_size': 16
        }
    elif model_type == 'swin':
        return {
            'name': 'Swin Transformer',
            'default_model': 'microsoft/swin-tiny-patch4-window7-224',
            'learning_rate': 2e-5,
            'batch_size': 16
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model on time series images')
    parser.add_argument('--model_type', type=str, default='vit', 
                        choices=['vit', 'efficientnet', 'convnext', 'swin'],
                        help='Type of model to use')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Specific model name (e.g., google/vit-base-patch16-224, b0)')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with labels')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (if not specified, use model-specific default)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (if not specified, use model-specific default)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone weights')
    parser.add_argument('--no_multi_layer', action='store_true',
                        help='Disable multi-layer feature fusion')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get model info
    model_info = get_model_info(args.model_type)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set model-specific parameters if not specified
    learning_rate = args.learning_rate or model_info['learning_rate']
    batch_size = args.batch_size or model_info['batch_size']
    model_name = args.model_name or model_info['default_model']
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and split data
    print(f"Loading data from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    
    # Save split data
    train_csv = os.path.join(args.output_dir, "train_labels.csv")
    val_csv = os.path.join(args.output_dir, "val_labels.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    # Create data loaders
    print(f"Creating data loaders with batch size {batch_size}")
    train_loader = create_image_data_loader(
        csv_path=train_csv,
        images_dir=args.images_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = create_image_data_loader(
        csv_path=val_csv,
        images_dir=args.images_dir,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print(f"Creating {model_info['name']} model: {model_name}")
    model = create_model(
        model_type=args.model_type,
        model_name=model_name,
        n_binary_targets=8,
        n_regression_targets=7,
        dropout=0.1,
        freeze_backbone=args.freeze_backbone,
        use_multi_layer_features=not args.no_multi_layer,
        pretrained=True
    )
    
    # Calculate class weights for binary targets
    binary_cols = [
        'Некачественное ГДИС', 'Влияние ствола скважины', 
        'Радиальный режим', 'Линейный режим', 'Билинейный режим', 
        'Сферический режим', 'Граница постоянного давления',
        'Граница непроницаемый разлом'
    ]
    
    pos_weights = []
    for col in binary_cols:
        # Calculate positive weight as ratio of negative to positive examples
        neg_count = (train_df[col] == 0).sum()
        pos_count = (train_df[col] == 1).sum()
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weights.append(weight)
    
    print("Positive weights for binary targets:")
    for col, weight in zip(binary_cols, pos_weights):
        print(f"  {col}: {weight:.2f}")
    
    # Train model
    print(f"\nTraining model for {args.num_epochs} epochs with learning rate {learning_rate}")
    print(f"Backbone {'frozen' if args.freeze_backbone else 'trainable'}, "
          f"Multi-layer features {'disabled' if args.no_multi_layer else 'enabled'}")
    
    trained_model, history = train_cv_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=learning_rate,
        device=device,
        binary_loss_weight=1.0,
        regression_loss_weight=0.1,
        pos_weight=pos_weights,
        warmup_epochs=1
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{args.model_type}_model.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save model configuration
    config = {
        'model_type': args.model_type,
        'model_name': model_name,
        'n_binary_targets': 8,
        'n_regression_targets': 7,
        'dropout': 0.1,
        'freeze_backbone': args.freeze_backbone,
        'use_multi_layer_features': not args.no_multi_layer
    }
    config_path = os.path.join(args.output_dir, f"{args.model_type}_config.json")
    with open(config_path, 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    # Plot training history
    history_path = os.path.join(args.output_dir, f"{args.model_type}_training_history.png")
    plot_training_history(history, history_path)
    print(f"Training history plot saved to {history_path}")
    
    # Run inference on validation set
    print("\nRunning inference on validation set...")
    predictions_path = os.path.join(args.output_dir, f"{args.model_type}_val_predictions.csv")
    save_cv_predictions(
        model=trained_model,
        dataloader=val_loader,
        output_path=predictions_path,
        device=device
    )
    print(f"Validation predictions saved to {predictions_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main() 