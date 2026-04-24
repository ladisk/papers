import copy
import torch
from torch import nn


# -------- Model components used for all datasets --------
class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor (SFE)
    
    Extract shared features from both source and target domains.

    input shape: (batch_size, input_dim)
    output shape: (batch_size, output_dim)

    parameters
    ----------
    input_dim: int
        The dimension of the input features.
    output_dim: int
        The dimension of the output features.
    dropout: float, optional
        The dropout rate (default: 0.1).
    
    structure
    ---------
    LinearLayer
    GELU
    Dropout
    LinearLayer

    For more detailed info check forward() method.
    """
    def __init__(self, input_dim:int, output_dim:int, dropout:float=0.1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)

class Classifier(nn.Module):
    """
    Classifier
    
    Classify the extracted features into the target classes.

    input shape: (batch_size, input_dim)
    output shape: (batch_size, num_classes) for multi-class classification
                 (batch_size, 1) for binary classification (logit)
    
    parameters
    ----------
    input_dim: int
        The dimension of the input features.
    num_classes: int
        The number of target classes. Must be at least 2.
    dropout: float, optional
        The dropout rate (default: 0.1).

    structure
    ---------
    LinearLayer
    GELU
    Dropout
    LinearLayer

    For more detailed info check forward() method.
    """
    def __init__(self, input_dim:int, num_classes:int, dropout:float=0.1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2 for binary classification")
        elif num_classes == 2:
            output_dim_size = 1 # For binary classification, output a single logit - use in combination with BCEWithLogitsLoss
        else:
            output_dim_size = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim_size)
        )

    def forward(self, x):
        return self.mlp(x)

class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) loss
    
    Calculates MMD loss between two sets of features to encourage domain-invariant feature learning.

    parameters
    ----------
    bandwidth_range: list of float, optional
        List of bandwidths for the multi-scale RBF kernel (default: [1, 2, 3]).

    For more detailed info check forward() method.
    """
    def __init__(self, bandwidth_range:list=[1, 2, 3]):
        super().__init__()
        self.bandwidth_range = bandwidth_range

    def gaussian_kernel(self, x, y):
        """
        Computes multi-scale RBF kernel between x and y.
        """
        n = x.size(0)
        m = y.size(0)

        x_norm = (x**2).sum(1).view(n, 1)
        y_norm = (y**2).sum(1).view(m, 1)

        dxy = x_norm + y_norm.T - 2 * torch.mm(x, y.T)

        K = 0
        for bw in self.bandwidth_range:
            K += torch.exp(-0.5 * dxy / bw)

        return K

    def forward(self, x, y):
        Kxx = self.gaussian_kernel(x, x)
        Kyy = self.gaussian_kernel(y, y)
        Kxy = self.gaussian_kernel(x, y)

        return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    
# -------- Model components used for time-series based dataset --------
class EmbeddingLayer(nn.Module):
    """
    Embedding layer for time-series data.

    Basic EmbeddingLayer without any overlap between patches. This function works only on 1D singals.
    Signal that is fed into this layer is of shape (1, d_signal). We need to make sure that
    d_signal % d_patch == 0.

    parameters
    ----------
    d_signal: int
        Dimension of the input signal.
    d_patch: int
        Dimension of patches.
    d_embedding: int
        Dimension of embedding vector.
    classification_token: bool, optional
        Whether to add a classification token at the beginning of the sequence (default: False).

    structure
    ---------
    LinearLayer

    For more detailed info check forward() method.
    """
    def __init__(self, d_signal:int, d_patch:int, d_embedding:int, classification_token:bool=False):
        super().__init__()
        self.d_signal = d_signal
        self.d_patch = d_patch
        self.d_embedding = d_embedding

        if self.d_signal % self.d_patch != 0:
            raise ValueError(f"Signal dimension {self.d_signal} must be divisible by patch dimension {self.d_patch}.")

        self.linear = nn.Linear(self.d_patch, self.d_embedding)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Classification token
        self.classification_token = classification_token
        if self.classification_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_embedding), requires_grad=True)

    def forward(self, x):
        batch_size, sequence_length = x.size(0), x.size(1)
        x = self._create_patches(x)
        x = self.linear(x)
        if self.classification_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        return x
    
    def _create_patches(self, x):
        """Splits the input signal into non-overlapping patches."""
        length_of_signal = x.size(1)
        if length_of_signal % self.d_patch != 0:
            raise ValueError(f"Signal length {length_of_signal} is not divisible by patch size {self.d_patch}.")
        return x.view(x.size(0), length_of_signal // self.d_patch, self.d_patch)
    
class RelativePositionBias(nn.Module):
    """
    Relative Position Bias for Transformer-based models.

    This module implements learnable relative position biases for self-attention mechanisms 
    in Transformer models. It allows the model to capture positional relationships between 
    tokens without relying on absolute positional encodings.

    parameters
    ----------
    n_heads: int
        The number of attention heads in the Transformer model.
    n_patch: int
        The number of patches in the input sequence. Calculated as sequence_length // patch_size (+ 1 if there's a classification token).
    classification_token: bool, optional
        Whether a classification token is included in the input sequence (default: False). If True, the bias will be adjusted to account for the CLS token.
    """
    def __init__(self, n_heads:int, n_patch:int, classification_token:bool=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_patch = n_patch
        self.classification_token = classification_token

        # Learnable parameters: one bias per head per relative distance
        self.bias_table = nn.Parameter(torch.zeros(self.n_heads, 2 * self.n_patch - 1))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def forward(self):
        pos = torch.arange(self.n_patch)
        rel = pos[:, None] - pos[None, :]
        rel += self.n_patch - 1
        rel = rel.clamp(0, 2 * self.n_patch - 2)

        bias = self.bias_table[:, rel]

        if self.classification_token:
            return bias

        # If CLS is used → zero its row and column
        # Assume CLS token is index 0
        bias = bias.clone()
        bias[:, 0, :] = 0
        bias[:, :, 0] = 0
        return bias

class MultiHeadSelfAttentionWithRPM(nn.Module):
    """
    Multi-Head Self-Attention with Relative Position Bias for Transformer models.
    
    parameters
    ----------
    d_embedding: int
        The dimension of the input embedding.
    n_heads: int
        The number of attention heads.
    n_patch: int
        The number of patches in the input sequence (including classification token if used).
    dropout: float, optional
        The dropout rate for attention weights (default: 0.1).
    classification_token: bool, optional
        Whether a classification token is included in the input sequence (default: False). If True, the attention mechanism will account for the CLS token in the relative position bias.
    """
    def __init__(self, d_embedding, n_heads, n_patch, dropout=0.1, classification_token:bool=False):
        super().__init__()
        self.d_embedding = d_embedding
        self.n_heads = n_heads
        self.n_patch = n_patch
        self.dropout = dropout
        self.classification_token = classification_token

        self.mha = nn.MultiheadAttention(d_embedding, n_heads, dropout=dropout, batch_first=True)
        self.rpb = RelativePositionBias(n_heads, n_patch, classification_token)
        
    def forward(self, x):
        # x: (batch, n_patch, d_embedding)
        batch_size = x.size(0)

        attn_mask = self.rpb()
        attn_mask = attn_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
        attn_mask = attn_mask.reshape(batch_size*self.n_heads, self.n_patch, self.n_patch)

        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out
    
class TransformerEncoderLayerRPB(nn.Module):
    """
    Transformer Encoder Layer

    parameters
    ----------
    d_embedding: int
        The dimension of the input embedding.
    n_heads: int
        The number of attention heads.
    n_patch: int
        The number of patches in the input sequence (including classification token if used).
    mlp_dim: int, optional
        The dimension of the MLP hidden layer (default: 256).
    dropout: float, optional
        The dropout rate for attention and MLP (default: 0.1).
    classification_token: bool, optional
        Whether a classification token is included in the input sequence (default: False).

    structure
    ---------
    Normalisation
    MultiHeadSelfAttention
    dropout + residual connection
    Normalisation
    MLP
    dropout + residual connection

    For more detailed info check forward() method.
    """
    def __init__(self, d_embedding:int, n_heads:int, n_patch:int, mlp_dim:int = 256, dropout=0.1, classification_token:bool=False):
        super().__init__()
        self.self_attn = MultiHeadSelfAttentionWithRPM(d_embedding, n_heads, n_patch, dropout, classification_token)
        self.norm1 = nn.LayerNorm(d_embedding)

        self.mpl = nn.Sequential(
            nn.Linear(d_embedding, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_embedding)
        )
        self.norm2 = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self attention + residual
        x = x + self.dropout(self.self_attn(self.norm1(x)))
        # MLP + residual
        x = x + self.dropout(self.mpl(self.norm2(x)))
        return x
    
class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple TransformerEncoderLayer layers.

    Copies the provided encoder layer n_layers times to create a stack of Transformer encoder layers.

    parameters
    ----------
    encoder_layer: nn.Module
        The Transformer encoder layer to be stacked.
    n_layers: int
        The number of encoder layers to stack.

    For more detailed info check forward() method.
    """
    def __init__(self, encoder_layer:nn.Module, n_layers:int):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.n_layers = n_layers
        self.encoder_layers = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.n_layers)])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class EncoderExtractionLayer(nn.Module):
    """
    Extraction layer for Transformer Encoder outputs.

    parameters
    ----------
    extraction: str
        Method of extraction. Options are:
        - "cls_token": Extract the classification token (first token).
        - "mean_pooling": Apply mean pooling over all tokens except the classification token.
        - "max_pooling": Apply max pooling over all tokens except the classification token.
    classification_token: bool
        Whether the input includes a classification token (default: False). If True, pooling methods (other than cls_token) will exclude the first token.

    For more detailed info check forward() method.
    """
    def __init__(self, extraction:str="cls_token", classification_token:bool=False):
        super().__init__()
        self.possible_extractions = ["cls_token", "mean_pooling", "max_pooling"]

        if extraction not in self.possible_extractions:
            raise ValueError(f"Extraction method {extraction} not recognized. Choose from {self.possible_extractions}.")
        self.extraction = extraction
        self.classification_token = classification_token

    def forward(self, x):
        if self.extraction == "cls_token":
            return x[:, 0, :]
        elif self.extraction == "mean_pooling":
            # Should exclude classification token if present
            if self.classification_token:
                return x[:, 1:, :].mean(dim=1)
            else:
                return x.mean(dim=1)
        elif self.extraction == "max_pooling":
            if self.classification_token:
                return x[:, 1:, :].max(dim=1).values
            else:
                return x.max(dim=1).values
