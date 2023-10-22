import os
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from torchvision.transforms import v2


class DFDataset(Dataset):
    def __init__(self, folder_path: str, label_path: str = None, transforms: Module = None) -> None:
        super().__init__()

        assert os.path.exists(folder_path)
        if label_path: assert os.path.exists(label_path)

        self.folder_path = folder_path
        self.label_path = label_path if label_path else folder_path 
        
        self.transform = v2.Identity() if not transforms else transforms

        self.images: list[Tensor] = self.load_images()
        self.labels: Tensor = self.load_labels()

        self.N = len(self.images)

    def load_images(self) -> list[Tensor]:
        images = list()

        for file in os.listdir(self.folder_path):
            if Path(file).suffix not in ['.jpg', '.png']: continue # TODO Get real list instead of placeholder
            file_path = os.path.join(self.folder_path, file)
            image =  np.array(Image.open(file_path))

            # Enforce channel dimension
            if len(image.shape) != 3:
                image = image.view(1, image.shape[0], image.shape[1])

            # Place channel into first dim
            if image.shape[0] not in [1, 3, 4]:
                image = image.transpose((2, 0, 1))

            images.append(torch.from_numpy(image))
        
        return images


    def load_labels(self) -> Tensor:

        # TODO Implement loader specific to given file type and/or dataset
        def load_labels_(path: str) -> Tensor:
            pass

        label_extensions = ['.txt', '.csv']
        
        if os.path.isdir(self.label_path):
            labels = []
            for file in os.listdir(self.label_path):
                if Path(file).suffix in label_extensions: # TODO Get real list
                    labels.append(file)
            
            if len(labels) != 1:
                raise RuntimeError(f'Found {len(labels)} possible label files in {self.label_path} ({labels}). Specify exact file.')
            
            return load_labels_(os.path.join(self.label_path, labels[0]))

        else:
            if Path(self.label_path).suffix not in label_extensions:
                raise RuntimeError(f'File does not contain labels. Possible extensions are {label_extensions}.')
            return load_labels_(self.label_path)


    def __getitem__(self, index: list[int]) -> tuple[Tensor]:
        return self.transform(self.images[index]), self.labels[index]

    def __len__(self) -> int:
        return self.N



def get_dataloader(path_images: str, path_labels:str = None, transforms: Module = None,
                   batch_size: int = 10, train: bool = True) -> DataLoader:
    return DataLoader(DFDataset(path_images, path_labels, transforms),
                      batch_size=batch_size, shuffle=train)