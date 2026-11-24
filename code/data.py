"""Dataset helpers for the assigned Histopathology image collection."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_paths(root):
    return sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class HistopathologyDataset(Dataset):
    def __init__(self, paths, image_size=128, augment=False):
        self.paths = list(paths)
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = ImageOps.fit(
                ImageOps.exif_transpose(source).convert("RGB"),
                (self.image_size, self.image_size),
                method=Image.Resampling.LANCZOS,
            )
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.augment:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
            if random.random() < 0.5:
                tensor = tensor.flip(-2)
            tensor = torch.rot90(tensor, random.randrange(4), dims=(-2, -1))
        return tensor
