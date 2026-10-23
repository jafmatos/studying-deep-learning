from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deep_learning.config import (
    BATCH_SIZE,
    DATA_DIR,
    TEST_DATA_DIR,
    TRAIN_DATA_DIR,
    VALIDATION_DATA_DIR,
)
from deep_learning.data.transforms import (
    test_transform,
    train_transform,
    validation_transform,
)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path = DATA_DIR, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.classes = sorted(
            [dir.name for dir in self.root_dir.iterdir() if dir.is_dir()]
        )
        self.class_to_index = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []

        for cls_name in self.classes:
            folder = self.root_dir / cls_name

            for img_path in folder.glob("*.*"):
                if img_path.suffix.lower() in (".jpg", ".png", ".jpeg"):
                    self.samples.append((img_path, self.class_to_index[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, label = self.samples[index]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
        

train_dataset = ImageFolderDataset(root_dir=TRAIN_DATA_DIR, transform=train_transform)
test_dataset = ImageFolderDataset(root_dir=TEST_DATA_DIR, transform=test_transform)
validation_dataset = ImageFolderDataset(root_dir=VALIDATION_DATA_DIR, transform=validation_transform)

train_data_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

validation_data_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)