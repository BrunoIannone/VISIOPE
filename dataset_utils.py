import utils
from PIL import Image
import os
from dataset_handler import DatasetHandler
from model import GameCartridgeDiscriminator
from pathlib import Path


# DATASET BUILD

dataset_handler = DatasetHandler(utils.DATA_PATH)
# dataset_handler.perform_sanity_check()
# utils.stack_and_resize_images2(
#     dataset_handler.samples, utils.STACKED_PATH
# )
print(dataset_handler.samples)
utils.build_couples(dataset_handler.samples)


# PREDICTION

# mod = GameCartridgeDiscriminator.load_from_checkpoint(
#     "/home/USER_NAME_HERE/Desktop/VISIOPE/Saves/ckpt/0.0001, 0.0001, 0, 0.01.ckpt"
# ).to("cpu")
# mod.eval()
# pred = mod.predict(utils.PREDICT_PATH)
# print(pred)
