"""
Target (Real) domain dataset with n-shot sampling support.
Samples n examples per class from the full target_train.csv based on seed.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import os
import random

# Default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_TRAIN_CSV = os.path.join(SCRIPT_DIR, "data_split", "train_real_split_0.05.csv")
TARGET_VAL_CSV = os.path.join(SCRIPT_DIR, "data_split", "val_real_split.csv")

BATCH_SIZE = 4  # Default, can be overridden


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class TargetDataset(Dataset):
    def __init__(self, csv_path, size=(128, 128), transforms_list=None):
        self.size = size
        df = pd.read_csv(csv_path)
        self.image_paths = list(df["path"])
        self.labels = list(df["label"])
        self.transforms = transforms_list if transforms_list else []

    def __getitem__(self, index):
        path = self.image_paths[index]
        label = torch.from_numpy(np.array(self.labels[index]))
        img = Image.open(path, 'r')
        video = []
        count = 0
        for i in range(5):
            for j in range(6):
                left = 29 * j
                top = 29 * i
                right = 29 * j + 28
                bottom = 29 * i + 28
                video.append(np.array(img.crop((left, top, right, bottom)).resize(self.size)))
                count += 1
                if count == 28:
                    break
            if count == 28:
                break
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


def create_nshot_csv(full_csv_path, n_shot, seed, output_csv_path=None):
    """
    Sample n examples per class from the full CSV based on seed.

    Args:
        full_csv_path: Path to full target_train.csv
        n_shot: Number of samples per class
        seed: Random seed for reproducibility
        output_csv_path: Optional path to save the n-shot CSV

    Returns:
        DataFrame with n samples per class
    """
    set_seed(seed)
    df = pd.read_csv(full_csv_path)

    # Group by label and sample n per class
    sampled_dfs = []
    for label in sorted(df['label'].unique()):
        class_df = df[df['label'] == label]
        if len(class_df) < n_shot:
            print(f"WARNING: Class {label} has only {len(class_df)} samples, but {n_shot}-shot requested!")
            sampled = class_df  # Use all available
        else:
            sampled = class_df.sample(n=n_shot, random_state=seed)
        sampled_dfs.append(sampled)

    nshot_df = pd.concat(sampled_dfs, ignore_index=True)

    # Optionally save to CSV
    if output_csv_path:
        nshot_df.to_csv(output_csv_path, index=False)
        print(f"Saved {n_shot}-shot CSV to {output_csv_path}")

    return nshot_df


def get_dataloaders(n_shot=3, seed=42, batch_size=None):
    """
    Get dataloaders for target domain with n-shot sampling.

    Args:
        n_shot: Number of samples per class for training
        seed: Random seed for sampling and dataloader
        batch_size: Batch size (uses global BATCH_SIZE if None)

    Returns:
        train_loader, val_loader, val_loader (test is same as val)
    """
    if batch_size is None:
        batch_size = BATCH_SIZE

    g = torch.Generator()
    g.manual_seed(seed)

    # Create n-shot training CSV in memory
    nshot_df = create_nshot_csv(TARGET_TRAIN_CSV, n_shot, seed)

    # Save to temporary CSV for dataset loading
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_csv_path = f.name
        nshot_df.to_csv(f, index=False)

    # Create datasets
    train_dataset = TargetDataset(temp_csv_path, size=(128, 128), transforms_list=[])
    val_dataset = TargetDataset(TARGET_VAL_CSV, size=(128, 128), transforms_list=[])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, pin_memory=True, generator=g, num_workers=0
    )

    # Clean up temp file
    os.unlink(temp_csv_path)

    print(f"Target domain: {len(train_dataset)} train samples ({n_shot}-shot), {len(val_dataset)} val samples")

    return train_loader, val_loader, val_loader
