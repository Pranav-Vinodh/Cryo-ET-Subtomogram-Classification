"""
Source (Simulated) domain dataset - uses the FULL source_train.csv.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import mrcfile
import os
import random

# Default paths (matches config_simulated_c10 — 0.25 subset)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_TRAIN_CSV = os.path.join(SCRIPT_DIR, "data_split_c10", "simulated_SNR005_train_0.25.csv")
SOURCE_VAL_CSV = os.path.join(SCRIPT_DIR, "data_split_c10", "simulated_SNR005_val.csv")

BATCH_SIZE = 4  # Default, can be overridden


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class SourceDataset(Dataset):
    def __init__(self, csv_path, size=(128, 128), transforms_list=None):
        self.size = size
        df = pd.read_csv(csv_path)
        self.image_paths = list(df["path"])
        self.labels = list(df["label"])
        self.transforms = transforms_list if transforms_list else []

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        mrc = mrcfile.open(path).data
        video = []
        for i in range(mrc.shape[0]):
            img = Image.fromarray(mrc[i])
            video.append(np.array(img.resize(self.size)))
        video = np.array(video)
        video = (video - video.min()) / (video.max() - video.min())
        video = np.stack((video,) * 3, axis=0)
        video = torch.from_numpy(video)
        for t in self.transforms:
            s = np.random.uniform()
            if s > 0.5:
                video = t(video)
        return video.float(), label

    def __len__(self):
        return len(self.labels)


def get_dataloaders(seed=42, batch_size=None):
    """
    Get dataloaders for source domain using FULL training data.

    Args:
        seed: Random seed for dataloader
        batch_size: Batch size (uses global BATCH_SIZE if None)

    Returns:
        train_loader, val_loader, val_loader (test is same as val)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    g = torch.Generator()
    g.manual_seed(seed)

    # Create datasets - use FULL source_train.csv
    train_dataset = SourceDataset(SOURCE_TRAIN_CSV, size=(128, 128), transforms_list=[])
    val_dataset = SourceDataset(SOURCE_VAL_CSV, size=(128, 128), transforms_list=[])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True, generator=g, num_workers=0
    )

    print(f"Source domain: {len(train_dataset)} train samples (FULL), {len(val_dataset)} val samples")

    return train_loader, val_loader, val_loader
