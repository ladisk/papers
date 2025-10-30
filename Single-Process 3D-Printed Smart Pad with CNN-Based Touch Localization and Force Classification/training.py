# Imports
import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, Tuple


class KendallLossWeights(nn.Module):
    """
    Learnable homoscedastic task uncertainties for multi-task learning.
    
    Combines location (L2) and force (CE) losses with learned weights:
      - Location: 0.5 * exp(-s_loc) * L2 + 0.5 * s_loc
      - Force: exp(-s_force) * CE + 0.5 * s_force
    
    Parameters s_loc and s_force are clamped to [-3, 3] for stability.
    """
    def __init__(self, init_log_vars: Tuple[float, float] = (0.0, 0.0)):
        super().__init__()
        s_loc, s_force = init_log_vars
        self.s_loc = nn.Parameter(torch.tensor(float(s_loc)))
        self.s_force = nn.Parameter(torch.tensor(float(s_force)))

    def forward(self, L_loc: torch.Tensor, L_force: torch.Tensor) -> torch.Tensor:
        """Compute weighted multi-task loss."""
        # Clamp for stability
        s_loc = torch.clamp(self.s_loc, -3.0, 3.0)
        s_force = torch.clamp(self.s_force, -3.0, 3.0)
        
        # Weighted losses
        loc_term = 0.5 * torch.exp(-s_loc) * L_loc + 0.5 * s_loc
        force_term = torch.exp(-s_force) * L_force + 0.5 * s_force
        
        return loc_term + force_term

    @property
    def weights(self) -> Dict[str, float]:
        """Return current task weights as dict."""
        s_loc = torch.clamp(self.s_loc, -3.0, 3.0)
        s_force = torch.clamp(self.s_force, -3.0, 3.0)
        return {
            "w_loc": float(torch.exp(-s_loc).detach().cpu()),
            "w_force": float(torch.exp(-s_force).detach().cpu()),
        }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    combiner: KendallLossWeights,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with loss, loc, force, and force_acc metrics.
    """
    model.train()
    loss_loc_fn = nn.MSELoss()
    loss_force_fn = nn.CrossEntropyLoss()
    
    stats = {"loss": 0.0, "loc": 0.0, "force": 0.0, "force_acc": 0.0, "n": 0}

    for batch in loader:
        X, loc_xy, force_y = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        pred_xy, force_logits = model(X)
        L_loc = loss_loc_fn(pred_xy, loc_xy)
        L_force = loss_force_fn(force_logits, force_y)
        
        loss = combiner(L_loc, L_force)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate stats
        B = X.size(0)
        stats["loss"] += loss.item() * B
        stats["loc"] += L_loc.item() * B
        stats["force"] += L_force.item() * B
        stats["force_acc"] += (force_logits.argmax(1) == force_y).sum().item()
        stats["n"] += B

    # Average over samples
    n = max(1, stats["n"])
    return {k: v / n for k, v in stats.items() if k != "n"}


def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    combiner: KendallLossWeights,
    device: torch.device
) -> Dict[str, float]:
    """
    Validate for one epoch.
    
    Returns:
        Dictionary with loss, loc, force, and force_acc metrics.
    """
    model.eval()
    loss_loc_fn = nn.MSELoss()
    loss_force_fn = nn.CrossEntropyLoss()
    
    stats = {"loss": 0.0, "loc": 0.0, "force": 0.0, "force_acc": 0.0, "n": 0}

    with torch.no_grad():
        for batch in loader:
            X, loc_xy, force_y = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            pred_xy, force_logits = model(X)
            L_loc = loss_loc_fn(pred_xy, loc_xy)
            L_force = loss_force_fn(force_logits, force_y)
            
            loss = combiner(L_loc, L_force)

            # Accumulate stats
            B = X.size(0)
            stats["loss"] += loss.item() * B
            stats["loc"] += L_loc.item() * B
            stats["force"] += L_force.item() * B
            stats["force_acc"] += (force_logits.argmax(1) == force_y).sum().item()
            stats["n"] += B

    # Average over samples
    n = max(1, stats["n"])
    return {k: v / n for k, v in stats.items() if k != "n"}


def train_model(
    model: nn.Module,
    loaders: Dict[str, torch.utils.data.DataLoader],
    max_epochs: int = 100,
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
    patience: int = 7,
    device: str = "cuda"
) -> Tuple[Dict[str, list], nn.Module]:
    """
    Train model with early stopping.
    
    Args:
        model: Neural network model
        loaders: Dict with 'train' and 'val' DataLoaders
        max_epochs: Maximum training epochs
        lr: Learning rate (constant)
        weight_decay: Weight decay for optimizer
        patience: Early stopping patience (epochs without improvement)
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        history: Dict with training metrics per epoch
        model: Trained model (best validation loss)
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup optimizer and loss combiner
    combiner = KendallLossWeights()
    all_params = list(model.parameters()) + list(combiner.parameters())
    optimizer = Adam(all_params, lr=lr, weight_decay=weight_decay)
    
    # Training history
    history = {
        "train_loss": [], "train_loc": [], "train_force": [], "train_acc": [],
        "val_loss": [], "val_loc": [], "val_force": [], "val_acc": []
    }
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"Training on {device}...")
    
    for epoch in range(max_epochs):
        # Train and validate
        train_stats = train_epoch(model, loaders["train"], optimizer, combiner, device)
        val_stats = validate_epoch(model, loaders["val"], combiner, device)
        
        # Store history
        history["train_loss"].append(train_stats["loss"])
        history["train_loc"].append(train_stats["loc"])
        history["train_force"].append(train_stats["force"])
        history["train_acc"].append(train_stats["force_acc"])
        history["val_loss"].append(val_stats["loss"])
        history["val_loc"].append(val_stats["loc"])
        history["val_force"].append(val_stats["force"])
        history["val_acc"].append(val_stats["force_acc"])
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            w = combiner.weights
            print(f"Epoch {epoch+1:3d}/{max_epochs} | "
                  f"Train: {train_stats['loss']:.4f} (loc:{train_stats['loc']:.4f}, "
                  f"force:{train_stats['force']:.4f}, acc:{train_stats['force_acc']:.3f}) | "
                  f"Val: {val_stats['loss']:.4f} (loc:{val_stats['loc']:.4f}, "
                  f"force:{val_stats['force']:.4f}, acc:{val_stats['force_acc']:.3f}) | "
                  f"w_loc={w['w_loc']:.3f}, w_force={w['w_force']:.3f}")
        
        # Early stopping
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history, model
