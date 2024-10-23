import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple
import os
from PIL import Image

class Elbit(Dataset):

    """
    num_classes: 17
    """
    CLASSES = ['Terrain', 'Unpaved Route', 'Paved Road', 'Tree Trunk', 'Tree Foliage', 'Rocks', 'Large Shrubs', 'Low Vegetation', 'Wire Fence',
                'Sky', 'Person', 'Vehicle', 'Building', 'ignore', 'Misc', 'Water', 'Animal']

    PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100]])

    ID2TRAINID = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, -1: 255}
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        #assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255
        self.root = root
        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        self.img_path = Path(self.root) / split
        #self.files = list(img_path.rglob('*.png'))

        image_file_names = [file for file in os.listdir(self.img_path) if file.endswith(
            '.png') and not file.endswith('_label.png')]
        mask_file_names = [f for f in os.listdir(self.img_path) if f.endswith('_label.png')]

        if not image_file_names:
            raise Exception(f"No images found in {self.img_path}")
        print(f"Found {len(image_file_names)} {split} images.")
        if not mask_file_names:
            raise Exception(f"No labels found in {self.img_path}")

        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        image = io.read_image(os.path.join(self.img_path, self.images[index]))
        label = io.read_image(os.path.join(self.img_path, self.masks[index]))

        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.squeeze().numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(Elbit, r'C:\eranandnadav\semantic-segmentation-main\data\elbit_dataset', 'train', batch_size=1)
