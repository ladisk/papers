import torch
from torch import nn


# -------- Model components used for all datasets --------
class SharedFeatureExtractor(nn.Module):
    """Shared feature extractor (SFE)
    
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
    """Classifier
    
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
    """ Maximum Mean Discrepancy (MMD) loss
    
    Calculates MMD loss between two sets of features to encourage domain-invariant feature learning.
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

# -------- Extracted features based dataset model -------- 
class ModelExtractedFeaturesDataset(nn.Module):
    def __init__(self, shared_feature_extractor:SharedFeatureExtractor, classifier:Classifier):
        super().__init__()
        self.shared_feature_extractor = shared_feature_extractor
        self.classifier = classifier

    def forward(self, x):
        extracted_features = self.shared_feature_extractor(x)
        logits = self.classifier(extracted_features)
        return logits
    
# -------- Model components used for time-series based dataset --------
class EmbeddingLayer(nn.Module):
    def __init__(self, d_patch:int, d_embedding:int, classification_token:bool=False):
        """
        Basic EmbeddingLayer without any overlap between patches. This function works only on 1D singals.
        Signal that is fed into this layer is of shape (1, d_signal). We need to make sure that
        d_signal // d_patch == 0.

        parameters
        ----------
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
        super().__init__()
        self.d_patch = d_patch
        self.d_embedding = d_embedding

        if self.d_patch // self.d_embedding != 0:
            raise ValueError(f"Patch dimension {self.d_patch} must be divisible by embedding dimension {self.d_embedding}.")

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

# -------- Time-series based dataset model --------
