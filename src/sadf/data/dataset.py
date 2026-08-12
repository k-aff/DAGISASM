"""PyTorch dataset over the manifest, plus loader/sampler builders.

Each item is (image_tensor, label_float, group_idx, row_position).
group_idx is -1 for unknown/ungrouped rows so losses can handle them.
"""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from ..config import DotDict

UNKNOWN_GROUP = -1


def build_group_vocab(df: pd.DataFrame, group_key: str, ordered: list[str]) -> list[str]:
    """Groups present in df, in configured order, excluding non-informative tags."""
    present = set(df[group_key].astype(str).unique()) if group_key in df else set()
    vocab = [g for g in ordered if g in present]
    vocab += sorted(present - set(vocab) - {"unknown", "n/a", "nan"})
    return vocab


def make_transforms(cfg: DotDict, train: bool) -> T.Compose:
    size = int(cfg.image.size)
    ops: list = [T.Resize((size, size))]
    if train:
        # Deliberately colour-preserving: photometric jitter would perturb the
        # very skin-tone signal whose fairness we are auditing.
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(cfg.image.mean, cfg.image.std)]
    return T.Compose(ops)


class ManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DotDict,
        group_vocab: list[str],
        group_key: str,
        train: bool,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.tf = make_transforms(cfg, train)
        self.group_to_idx = {g: i for i, g in enumerate(group_vocab)}
        self.group_key = group_key

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        x = self.tf(img)
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        g = self.group_to_idx.get(str(row.get(self.group_key, "unknown")), UNKNOWN_GROUP)
        return x, y, torch.tensor(g, dtype=torch.long), torch.tensor(i, dtype=torch.long)


def make_balanced_sampler(
    df: pd.DataFrame, group_key: str
) -> WeightedRandomSampler:
    """Inverse-frequency sampling over (group x label) cells."""
    cells = df[group_key].astype(str) + "|" + df["label"].astype(str)
    counts = cells.value_counts()
    weights = cells.map(lambda c: 1.0 / counts[c]).to_numpy().copy()
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights),
        replacement=True,
    )


def make_loader(
    df: pd.DataFrame,
    cfg: DotDict,
    group_vocab: list[str],
    train: bool,
    balanced: bool = False,
) -> DataLoader:
    ds = ManifestDataset(df, cfg, group_vocab, cfg.group.key, train)
    sampler = make_balanced_sampler(df, cfg.group.key) if (train and balanced) else None
    return DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=int(cfg.train.num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
