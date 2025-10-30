# Imports
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader, random_split


class ImpactDataset(Dataset):
    """    
    Each sample contains:
    - x: Sensor time series (T, num_sensors)
    - loc_xy: Impact location (x, y) coordinates
    - force_cls: Force magnitude class (0=low, 1=medium, 2=high)
    
    Args:
        data_dir: Directory containing .pkl files with impact data
        force_bounds: Tuple (b1, b2) defining class thresholds:
            - force < b1: class 0
            - b1 <= force < b2: class 1
            - force >= b2: class 2
        remove_sensors: Optional tuple of sensor indices to exclude (0-indexed)
    
    Example:
        dataset = ImpactDataset(
            data_dir='data/1000Hz',
            force_bounds=(2, 3),
            remove_sensors=(1, 3)  # Keep only sensors 0 and 2
        )
    """
    def __init__(
        self, 
        data_dir: str, 
        force_bounds: Tuple[float, float] = (2, 3),
        remove_sensors: Optional[Tuple[int, ...]] = None
    ):
        super().__init__()
        self.samples = []
        self.remove_sensors = remove_sensors
        b1, b2 = force_bounds
        
        data_dir = Path(data_dir)
        
        # Load all .pkl files in directory
        for pkl_path in sorted(data_dir.glob("*.pkl")):
            with open(pkl_path, "rb") as f:
                data_dict = pickle.load(f)
            
            # Location is same for all repeats in file
            x_coord, y_coord = map(float, data_dict["label_loc"])
            
            # Time-series data: list of (T, 4) arrays
            data_list = data_dict.get("data", [])
            
            # Per-sample measured force traces
            force_list = data_dict.get("measured_force", None)
            
            for i, sensor_array in enumerate(data_list):
                # --- Build input tensor (T, num_sensors) ---
                arr = np.asarray(sensor_array, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[:, None]  # (T,) -> (T, 1)
                
                # Apply sensor removal if specified
                if remove_sensors is not None:
                    all_sensors = set(range(arr.shape[1]))
                    sensors_to_remove = set(remove_sensors)
                    sensors_to_keep = sorted(all_sensors - sensors_to_remove)
                    
                    if len(sensors_to_keep) == 0:
                        raise ValueError("Cannot remove all sensors!")
                    
                    arr = arr[:, sensors_to_keep]
                
                x_tensor = torch.from_numpy(arr)  # (T, num_sensors)
                
                # --- Extract maximum force for this sample ---
                F_max = None
                if force_list is not None and len(force_list) > i and force_list[i] is not None:
                    F = np.asarray(force_list[i], dtype=np.float32)
                    F_max = float(np.max(np.abs(F)))
                elif "hammer_max_force" in data_dict:
                    # Fallback: use scalar value from metadata
                    val = data_dict["hammer_max_force"]
                    if isinstance(val, (list, tuple)) and len(val) > i:
                        F_max = float(val[i])
                    else:
                        F_max = float(val)
                
                # --- Map force to class using thresholds ---
                if F_max < b1:
                    force_class = 0
                elif F_max < b2:
                    force_class = 1
                else:
                    force_class = 2
                
                # Store sample
                loc_xy = torch.tensor([x_coord, y_coord], dtype=torch.float32)
                force_cls = torch.tensor(force_class, dtype=torch.long)
                self.samples.append((x_tensor, loc_xy, force_cls))
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            x: Sensor time series (T, num_sensors)
            loc_xy: Location coordinates (2,)
            force_cls: Force class label (scalar)
        """
        return self.samples[index]
    
    def get_num_sensors(self) -> int:
        """Return number of active sensors after removal."""
        if len(self.samples) > 0:
            return self.samples[0][0].shape[1]
        return 0


def ImpactDataloader(
    dataset: Dataset,
    batch_size: int = 32,
    seed: int = 42
) -> dict:
    """
    Create train/val/test DataLoaders with 80/10/10 split.
    
    Args:
        dataset: ImpactDataset instance
        batch_size: Batch size for all loaders
        seed: Random seed for reproducible splits
    
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing DataLoader objects
    
    Example:
        dataset = ImpactDataset('data/1000Hz')
        loaders = ImpactDataloader(dataset, batch_size=32)
        
        for batch in loaders['train']:
            x, loc_xy, force_cls = batch
            # x: (B, T, num_sensors)
            # loc_xy: (B, 2)
            # force_cls: (B,)
    """
    N = len(dataset)
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val
    
    # Create splits with fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [n_train, n_val, n_test], 
        generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
