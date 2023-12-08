import os
from utils import DATA_PATH, show_dataset
from dataset_handler import DatasetHandler
from termcolor import colored
from PIL import Image
from pathlib import Path
import cv2


root_folder = Path(DATA_PATH)

dataset_handler = DatasetHandler(root_folder, 1, 2, 3)
dataset_handler.perform_sanity_check()
# print(dataset_handler.training_couples)
# show_dataset(dataset_handler.training_couples)
