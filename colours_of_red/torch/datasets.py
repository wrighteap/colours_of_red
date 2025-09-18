from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import cv2
import torch
from colours_of_red.config import PROCESSED_DATA_DIR

class RaspberrySet(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        super().__init__()

        self.transform = transform

        with open(root_dir / 'classes.txt', 'r') as file:
            self.idx_to_class = {int(line.split()[0]): ' '.join(line.split()[1:]) for line in file}

        metadata = []
        for filepath in list(root_dir.rglob("*.JPEG")):
            df = pd.read_csv(
                root_dir / f"{filepath.stem}.txt", names=["class_idx", "center_x", "center_y", "width", "height"],
                sep=r"\s+", header = None
            )
            df["filepath"] = str(filepath)
            metadata.append(df)

        self.metadata = pd.concat(metadata).reset_index(drop=True).astype(
            {
                "filepath": "str",
                "class_idx": "int",
                "center_x": "float",
                "center_y": "float",
                "width": "float",
                "height": "float"
            }
        )

    def load_image_to_tensor(self, filepath: str):
        image = cv2.imread(filepath)  # Loads in BGR
        if image is None:
            raise FileNotFoundError(f"Image {filepath} not found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def __getitem__(self, index):
        image = self.load_image_to_tensor(self.metadata.loc[index, "filepath"])
        class_idx = self.metadata.loc[index, "class_idx"]

        if self.transform:
            image = self.transform(image)

        return {
            "inputs": image,
            "targets": class_idx,
            "idx": index
        }
    
    def __len__(self):
        return len(self.metadata)