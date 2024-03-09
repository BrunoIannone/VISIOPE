from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from typing import List
import time
from torchvision.transforms import v2
import torch


class GameCartridgeDiscriminatorDataset(Dataset):
    """Game cartridge dataset class"""

    def __init__(self, samples: List[tuple]):
        """Constructor for cartridge dataset dataset

        Args:
            samples (List[tuple]): List of tuples (image_path, action_label)

        """

        self.samples = samples

        self.labels_to_idx = {
            "DS": 0,
            "GBA": 1,
            "GB": 2,
        }

        self.originality_labels_to_idx = {
            "true": 3,
            "false": 4,
        }

    def __len__(self):
        """Return samples length

        Returns:
            int: length of samples list (number of samples)
        """
        return len(self.samples)

    def __getitem__(self, index: int):
        """Get item for dataloader input

        Args:
            index (int): index-th sample to access

        Returns:
            tuple: (image, labels) open image_path and return the tuple (image,label) related to the index-th element
        """

        resize_transform = v2.Compose(
            [
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        # print(self.samples[index][0].split("£")[0])
        # print(self.samples[index][0].split("£")[1])
        # time.sleep(5)

        front = resize_transform(
            read_image(self.samples[index][0].split("£")[0], mode=ImageReadMode.RGB)
        )
        # print(front.shape)
        rear = resize_transform(
            read_image(self.samples[index][0].split("£")[1], mode=ImageReadMode.RGB)
        )
        # print(rear.shape)
        label_binary = self.build_binary_label_vector(index)
        # time.sleep(2)

        return front, rear, torch.tensor(label_binary, dtype=torch.float32)

    def build_binary_label_vector(self, index: int):
        """Return the label corresponding to a binary vector

        Args:
            index (int): Sample index

        Returns:
            List[int]: Label binary vector
        """

        label = self.samples[index][1]  # get label

        label = label.split(" ")  # [console, {true,false}]
        # print(label)

        cartridge = self.labels_to_idx[label[0]]
        label_binary = [
            0,
            0,
            0,
            0,
            0,
        ]
        label_binary[cartridge] = 1  # set one in the console index
        value = self.originality_labels_to_idx[
            label[1]
        ]  # set one in the true/false index

        label_binary[value] = 1
        return label_binary
