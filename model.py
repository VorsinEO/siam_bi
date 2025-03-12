import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        """
        Positional encoding for transformer model.
        
        Args:
            d_model: Dimension of the model
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and store
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_binary_targets: int = 8,
        n_regression_targets: int = 7,
        max_seq_len: int = 512
    ):
        """
        Initialize the transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            n_binary_targets: Number of binary targets
            n_regression_targets: Number of regression targets
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Store configuration for from_pretrained
        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'n_binary_targets': n_binary_targets,
            'n_regression_targets': n_regression_targets,
            'max_seq_len': max_seq_len
        }
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.binary_output = nn.Linear(d_model, n_binary_targets)  # No sigmoid, we'll use BCEWithLogitsLoss
        self.regression_output = nn.Linear(d_model, n_regression_targets)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (binary_logits, regression_output)
            - binary_logits: Binary classification logits (not probabilities)
            - regression_output: Regression predictions
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create src_key_padding_mask for transformer
        # PyTorch transformer expects padding mask where True indicates position to mask
        src_key_padding_mask = ~attention_mask.bool()
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling over sequence dimension
        # Only consider non-padded elements (where mask is True)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
        masked_x = x * mask_expanded
        
        # Sum and divide by the number of non-padded elements
        sum_x = masked_x.sum(dim=1)
        count_non_padded = attention_mask.sum(dim=1, keepdim=True)
        pooled_x = sum_x / (count_non_padded + 1e-10)
        
        # Pass through the output layers
        binary_logits = self.binary_output(pooled_x)  # No sigmoid here, return logits
        regression_output = self.regression_output(pooled_x)
        
        return binary_logits, regression_output 

    @classmethod
    def from_pretrained(cls, weights_path: str, eval_mode: bool = True, **kwargs):
        """
        Create a model and load pretrained weights.
        
        Args:
            weights_path: Path to the saved weights
            eval_mode: Whether to set the model to evaluation mode
            **kwargs: Override default model configuration
            
        Returns:
            Model with loaded weights
        """
        # Load the state dict
        state_dict = torch.load(weights_path)
        
        # Create new model instance
        if 'config' in state_dict:
            # Use saved configuration
            config = state_dict['config']
            # Override with any provided kwargs
            config.update(kwargs)
            model = cls(**config)
            # Load only model weights
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            # If no config in state dict, use default/provided args
            model = cls(**kwargs)
            # Load weights directly
            model.load_state_dict(state_dict)
        
        if eval_mode:
            model.eval()
        
        return model

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) implementation.
    Based on the paper: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create frequency bands
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, freqs)  # [seq_len, dim/2]
        
        # Create rotation matrices
        cos = torch.cos(freqs)  # [seq_len, dim/2]
        sin = torch.sin(freqs)  # [seq_len, dim/2]
        
        # Register buffers for cos and sin
        self.register_buffer('cos', cos)  # [seq_len, dim/2]
        self.register_buffer('sin', sin)  # [seq_len, dim/2]
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tuple of (rotated_q, rotated_k) with rotary position embeddings applied
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Get cos and sin for current sequence length
        cos = self.cos[:seq_len, :(head_dim//2)]  # [seq_len, head_dim/2]
        sin = self.sin[:seq_len, :(head_dim//2)]  # [seq_len, head_dim/2]
        
        # Reshape for broadcasting
        # [seq_len, head_dim/2] -> [1, 1, seq_len, head_dim/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Split head_dim into two halves
        q_left, q_right = q[..., :head_dim//2], q[..., head_dim//2:head_dim]
        k_left, k_right = k[..., :head_dim//2], k[..., head_dim//2:head_dim]
        
        # Apply rotary embeddings
        q_rotated_left = q_left * cos - q_right * sin
        q_rotated_right = q_left * sin + q_right * cos
        k_rotated_left = k_left * cos - k_right * sin
        k_rotated_right = k_left * sin + k_right * cos
        
        # Concatenate back
        q_rotated = torch.cat([q_rotated_left, q_rotated_right], dim=-1)
        k_rotated = torch.cat([k_rotated_left, k_rotated_right], dim=-1)
        
        return q_rotated, k_rotated


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    Based on the paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * (x / rms)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function
    Based on the paper: https://arxiv.org/abs/2002.05202
    """
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU activation to input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after SwiGLU activation
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2  # SwiGLU activation
        return self.w3(hidden)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with rotary positional embeddings
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Rotary positional embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention with rotary positional embeddings.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            key_padding_mask: Boolean mask where True indicates padding (to be masked)
            
        Returns:
            Output tensor after attention
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to queries, keys, values
        q = self.q_proj(x)  # [batch_size, seq_len, dim]
        k = self.k_proj(x)  # [batch_size, seq_len, dim]
        v = self.v_proj(x)  # [batch_size, seq_len, dim]
        
        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Apply rotary positional embeddings
        q, k = self.rope(q, k)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # Convert mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape and project back
        output = output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Modern transformer block with SwiGLU and RMSNorm
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout, max_seq_len)
        self.norm2 = RMSNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, hidden_features, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer block to input tensor.
        
        Args:
            x: Input tensor
            key_padding_mask: Boolean mask where True indicates padding
            
        Returns:
            Output tensor after transformer block
        """
        # Pre-norm, attention, and residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, key_padding_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Pre-norm, MLP, and residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = residual + x
        
        return x


class TransformerModel_ver2(nn.Module):
    """
    Modern transformer model with rotary positional embeddings, SwiGLU, and RMSNorm
    """
    def __init__(
        self,
        input_dim: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        n_binary_targets: int = 8,
        n_regression_targets: int = 7,
        max_seq_len: int = 512
    ):
        """
        Initialize the modern transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Dimension of the model
            nhead: Number of heads in multi-head attention
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            n_binary_targets: Number of binary targets
            n_regression_targets: Number of regression targets
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        # Store configuration for from_pretrained
        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_encoder_layers': num_encoder_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'n_binary_targets': n_binary_targets,
            'n_regression_targets': n_regression_targets,
            'max_seq_len': max_seq_len
        }
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=d_model,
                num_heads=nhead,
                mlp_ratio=dim_feedforward / d_model,
                dropout=dropout,
                max_seq_len=max_seq_len
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(d_model)
        
        # Output layers
        self.binary_output = nn.Linear(d_model, n_binary_targets)
        self.regression_output = nn.Linear(d_model, n_regression_targets)
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (binary_logits, regression_output)
            - binary_logits: Binary classification logits (not probabilities)
            - regression_output: Regression predictions
        """
        # Input projection
        x = self.input_projection(x)
        
        # Create key padding mask (True indicates positions to mask)
        key_padding_mask = ~attention_mask.bool()
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, key_padding_mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Global average pooling over sequence dimension
        # Only consider non-padded elements (where mask is True)
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
        masked_x = x * mask_expanded
        
        # Sum and divide by the number of non-padded elements
        sum_x = masked_x.sum(dim=1)
        count_non_padded = attention_mask.sum(dim=1, keepdim=True)
        pooled_x = sum_x / (count_non_padded + 1e-10)
        
        # Pass through the output layers
        binary_logits = self.binary_output(pooled_x)
        regression_output = self.regression_output(pooled_x)
        
        return binary_logits, regression_output 

    @classmethod
    def from_pretrained(cls, weights_path: str, eval_mode: bool = True, **kwargs):
        """
        Create a model and load pretrained weights.
        
        Args:
            weights_path: Path to the saved weights
            eval_mode: Whether to set the model to evaluation mode
            **kwargs: Override default model configuration
            
        Returns:
            Model with loaded weights
        """
        # Load the state dict
        state_dict = torch.load(weights_path)
        
        # Create new model instance
        if 'config' in state_dict:
            # Use saved configuration
            config = state_dict['config']
            # Override with any provided kwargs
            config.update(kwargs)
            model = cls(**config)
            # Load only model weights
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            # If no config in state dict, use default/provided args
            model = cls(**kwargs)
            # Load weights directly
            model.load_state_dict(state_dict)
        
        if eval_mode:
            model.eval()
        
        return model 