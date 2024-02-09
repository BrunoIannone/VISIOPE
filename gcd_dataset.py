from torch.utils.data import Dataset

# import stud.transformer_utils as transformer_utils
from torchvision.io import read_image
from termcolor import colored
from typing import List
import time
import os
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import cv2
import utils
import numpy as np
import torch


class GameCartridgeDiscriminatorDataset(Dataset):
    """Car action dataset class"""

    def __init__(self, samples: List[tuple]):
        """Constructor for car action dataset

        Args:
            samples (List[tuple]): List of tuples (image_path, action_label)

        """

        self.samples = samples
        # print(self.samples)
        # self.labels_to_idx = {
        #     "DS true": 0,
        #     "DS false": 1,
        #     "GBA true": 2,
        #     "GBA false": 3,
        #     "GB true": 4,
        #     "GB false": 5,
        # }
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
        target_size = (2000, 2000)

        # convert index-th sample senses in indices
        resize_transform = v2.Compose(
            [
                v2.Resize(target_size, antialias=True),
                v2.ToDtype(
                    torch.float32, scale=True
                ),  # If you want to convert the PIL Image to a PyTorch tensor
            ]
        )
        image = resize_transform(read_image(self.samples[index][0]))
        label = self.samples[index][1]
        label = label.split(" ")
        # print(label)
        cartridge = self.labels_to_idx[label[0]]
        label_binary = [
            0,
            0,
            0,
            0,
            0,
        ]
        label_binary[cartridge] = 1
        value = self.originality_labels_to_idx[label[1]]

        label_binary[value] = 1

        # print(torch.tensor(label_binary, dtype=torch.float32))
        return image, torch.tensor(label_binary, dtype=torch.float32)
