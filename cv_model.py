import torch
import torch.nn as nn
from typing import Tuple, Optional, Union, Dict, Any
from transformers import (
    ViTModel, ViTConfig,
    SwinModel, SwinConfig,
    ConvNextModel, ConvNextConfig,
    AutoModel, AutoConfig
)
from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, 
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights, 
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights
)


class TimeSeriesImageModel(nn.Module):
    """
    Base class for time series image models with different backbones.
    """
    def __init__(
        self,
        backbone_type: str = "vit",
        backbone_model: Optional[Union[str, nn.Module]] = None,
        n_binary_targets: int = 8,
        n_regression_targets: int = 7,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_multi_layer_features: bool = True,
        pretrained: bool = True
    ):
        """
        Initialize the model.
        
        Args:
            backbone_type: Type of backbone model ('vit', 'efficientnet', 'convnext', 'swin')
            backbone_model: Specific model name or instance (e.g., 'google/vit-base-patch16-224')
            n_binary_targets: Number of binary targets
            n_regression_targets: Number of regression targets
            dropout: Dropout rate
            freeze_backbone: Whether to freeze the backbone
            use_multi_layer_features: Whether to use features from multiple layers
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Initialize backbone and get hidden size
        self.backbone_type = backbone_type.lower()
        self.backbone, self.hidden_size = self._init_backbone(
            backbone_type, backbone_model, pretrained
        )
        self.use_multi_layer_features = use_multi_layer_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(self.hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature fusion (if using multi-layer features)
        if use_multi_layer_features and backbone_type in ['vit', 'swin']:
            # We'll use the last 4 layers
            self.feature_fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 4, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Binary classification head
        self.binary_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, n_binary_targets)
        )
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, n_regression_targets)
        )
    
    def _init_backbone(
        self, 
        backbone_type: str, 
        backbone_model: Optional[Union[str, nn.Module]],
        pretrained: bool
    ) -> Tuple[nn.Module, int]:
        """
        Initialize the backbone model.
        
        Args:
            backbone_type: Type of backbone model
            backbone_model: Specific model name or instance
            pretrained: Whether to use pretrained weights
            
        Returns:
            Tuple of (backbone_model, hidden_size)
        """
        if backbone_type == 'vit':
            # Vision Transformer
            if isinstance(backbone_model, nn.Module):
                backbone = backbone_model
                hidden_size = backbone.config.hidden_size
            else:
                model_name = backbone_model or "google/vit-base-patch16-224"
                config = ViTConfig.from_pretrained(model_name, output_hidden_states=True)
                backbone = ViTModel.from_pretrained(model_name, config=config)
                hidden_size = backbone.config.hidden_size
            
            return backbone, hidden_size
            
        elif backbone_type == 'efficientnet':
            # EfficientNet
            if isinstance(backbone_model, nn.Module):
                backbone = backbone_model
                # Remove the classifier
                backbone = nn.Sequential(*list(backbone.children())[:-1])
                # Get hidden size from the last layer
                hidden_size = backbone[-1][-1].out_channels
            else:
                model_name = backbone_model or "b0"
                if model_name == "b0":
                    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b0(weights=weights)
                    hidden_size = 1280
                elif model_name == "b1":
                    weights = EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b1(weights=weights)
                    hidden_size = 1280
                elif model_name == "b2":
                    weights = EfficientNet_B2_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b2(weights=weights)
                    hidden_size = 1408
                elif model_name == "b3":
                    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b3(weights=weights)
                    hidden_size = 1536
                elif model_name == "b4":
                    weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b4(weights=weights)
                    hidden_size = 1792
                elif model_name == "b5":
                    weights = EfficientNet_B5_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b5(weights=weights)
                    hidden_size = 2048
                elif model_name == "b6":
                    weights = EfficientNet_B6_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b6(weights=weights)
                    hidden_size = 2304
                elif model_name == "b7":
                    weights = EfficientNet_B7_Weights.IMAGENET1K_V1 if pretrained else None
                    backbone = efficientnet_b7(weights=weights)
                    hidden_size = 2560
                else:
                    raise ValueError(f"Unknown EfficientNet model: {model_name}")
                
                # Remove the classifier
                backbone = nn.Sequential(*list(backbone.children())[:-1])
            
            # Add adaptive pooling to ensure fixed output size
            backbone = nn.Sequential(
                backbone,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            
            return backbone, hidden_size
            
        elif backbone_type == 'convnext':
            # ConvNeXt
            if isinstance(backbone_model, nn.Module):
                backbone = backbone_model
                hidden_size = backbone.config.hidden_sizes[-1]
            else:
                model_name = backbone_model or "facebook/convnext-tiny-224"
                config = ConvNextConfig.from_pretrained(model_name, output_hidden_states=True)
                backbone = ConvNextModel.from_pretrained(model_name, config=config)
                hidden_size = backbone.config.hidden_sizes[-1]
            
            return backbone, hidden_size
            
        elif backbone_type == 'swin':
            # Swin Transformer
            if isinstance(backbone_model, nn.Module):
                backbone = backbone_model
                hidden_size = backbone.config.hidden_size
            else:
                model_name = backbone_model or "microsoft/swin-tiny-patch4-window7-224"
                config = SwinConfig.from_pretrained(model_name, output_hidden_states=True)
                backbone = SwinModel.from_pretrained(model_name, config=config)
                hidden_size = backbone.config.hidden_size
            
            return backbone, hidden_size
            
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            images: Input images of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (binary_logits, regression_output)
            - binary_logits: Binary classification logits (not probabilities)
            - regression_output: Regression predictions
        """
        # Extract features from backbone
        if self.backbone_type == 'vit':
            outputs = self.backbone(pixel_values=images, output_hidden_states=True)
            
            if self.use_multi_layer_features:
                # Get features from last 4 layers
                hidden_states = outputs.hidden_states[-4:]
                
                # Extract CLS tokens from each layer
                cls_tokens = [hs[:, 0] for hs in hidden_states]
                
                # Concatenate along feature dimension
                multi_layer_features = torch.cat(cls_tokens, dim=1)
                
                # Apply feature fusion
                features = self.feature_fusion(multi_layer_features)
            else:
                # Get CLS token from last layer
                features = outputs.last_hidden_state[:, 0]
                
        elif self.backbone_type == 'efficientnet':
            # EfficientNet outputs a flattened feature vector
            features = self.backbone(images)
            
        elif self.backbone_type == 'convnext':
            outputs = self.backbone(pixel_values=images, output_hidden_states=True)
            
            # Get pooled output
            features = outputs.pooler_output
            
        elif self.backbone_type == 'swin':
            outputs = self.backbone(pixel_values=images, output_hidden_states=True)
            
            if self.use_multi_layer_features:
                # Get features from last 4 layers
                hidden_states = outputs.hidden_states[-4:]
                
                # Get pooled outputs from each layer
                pooled_features = []
                for hs in hidden_states:
                    # Global average pooling
                    pooled = torch.mean(hs, dim=1)
                    pooled_features.append(pooled)
                
                # Concatenate along feature dimension
                multi_layer_features = torch.cat(pooled_features, dim=1)
                
                # Apply feature fusion
                features = self.feature_fusion(multi_layer_features)
            else:
                # Global average pooling on the last hidden state
                features = torch.mean(outputs.last_hidden_state, dim=1)
        
        # Pass through output heads
        binary_logits = self.binary_head(features)
        regression_output = self.regression_head(features)
        
        return binary_logits, regression_output


# Legacy class names for backward compatibility
class ViTTimeSeriesModel(TimeSeriesImageModel):
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        n_binary_targets: int = 8,
        n_regression_targets: int = 7,
        dropout: float = 0.1,
        freeze_backbone: bool = False
    ):
        super().__init__(
            backbone_type="vit",
            backbone_model=pretrained_model_name,
            n_binary_targets=n_binary_targets,
            n_regression_targets=n_regression_targets,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_multi_layer_features=False
        )


class ViTTimeSeriesModel_v2(TimeSeriesImageModel):
    def __init__(
        self,
        pretrained_model_name: str = "google/vit-base-patch16-224",
        n_binary_targets: int = 8,
        n_regression_targets: int = 7,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_multi_layer_features: bool = True
    ):
        super().__init__(
            backbone_type="vit",
            backbone_model=pretrained_model_name,
            n_binary_targets=n_binary_targets,
            n_regression_targets=n_regression_targets,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_multi_layer_features=use_multi_layer_features
        )


def create_model(
    model_type: str = "vit",
    model_name: Optional[str] = None,
    n_binary_targets: int = 8,
    n_regression_targets: int = 7,
    dropout: float = 0.1,
    freeze_backbone: bool = False,
    use_multi_layer_features: bool = True,
    pretrained: bool = True,
    weights_path: Optional[str] = None
) -> TimeSeriesImageModel:
    """
    Factory function to create a model with the specified backbone.
    
    Args:
        model_type: Type of model ('vit', 'efficientnet', 'convnext', 'swin')
        model_name: Specific model name (e.g., 'google/vit-base-patch16-224', 'b0' for EfficientNet)
        n_binary_targets: Number of binary targets
        n_regression_targets: Number of regression targets
        dropout: Dropout rate
        freeze_backbone: Whether to freeze the backbone
        use_multi_layer_features: Whether to use features from multiple layers
        pretrained: Whether to use pretrained weights
        weights_path: Optional path to load model weights from
        
    Returns:
        Initialized model
    """
    # Set default model names if not provided
    if model_name is None:
        if model_type == 'vit':
            model_name = "google/vit-base-patch16-224"
        elif model_type == 'efficientnet':
            model_name = "b0"
        elif model_type == 'convnext':
            model_name = "facebook/convnext-tiny-224"
        elif model_type == 'swin':
            model_name = "microsoft/swin-tiny-patch4-window7-224"
    
    model = TimeSeriesImageModel(
        backbone_type=model_type,
        backbone_model=model_name,
        n_binary_targets=n_binary_targets,
        n_regression_targets=n_regression_targets,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        use_multi_layer_features=use_multi_layer_features,
        pretrained=pretrained
    )
    
    # Load weights if path is provided
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
        model.eval()  # Set to evaluation mode
    
    return model 