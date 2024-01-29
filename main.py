import os
from utils import (
    PATH,
    DATA_PATH,
    STACKED_PATH,
    ROBO_PATH,
    TRAINING_PATH,
    VALID_PATH,
    TEST_PATH,
    show_dataset,
    stack_and_resize_images,
)
from dataset_handler import DatasetHandler
from termcolor import colored
from PIL import Image
from pathlib import Path
import cv2
import csv
import utils

labels_to_idx = {"DS": 1, "GB": 2, "GBA": 3, "true": 4, "false": 5}
idx_to_labels = {1: "DS", 2: "GB", 3: "GBA", 4: "true", 5: "false"}

root_folder = Path(ROBO_PATH)
dest_folder = Path(STACKED_PATH)
dataset_handler = DatasetHandler(
    root_folder, Path(TRAINING_PATH), Path(VALID_PATH), Path(TEST_PATH)
)
# dataset_handler.perform_sanity_check()
# stack_and_resize_images(dataset_handler.training_couples,dest_folder,(1000,1000))
# robo = Path(ROBO_PATH + "/train")
# print(dataset_handler.eval_samples)
