import utils
from PIL import Image
import os
from dataset_handler import DatasetHandler
from model import GameCartridgeDiscriminator
from pathlib import Path

# utils.manipola_immagine(utils.ROOT_FOOLDER/"Data/test/4",[(None, "_crop")],utils.ROOT_FOOLDER/"Data_crop/test/4")
# train_samples = utils.build_couples(utils.STACKED_PATH)
# train_samples = utils.build_couples(Path("/home/bruno/Desktop/VISIOPE/Predict/"))

# for file in os.listdir(utils.TEST_DIR_PATH / "0"):
#     utils.draw_random_black_square_on_image(
#         utils.TEST_DIR_PATH / "0" / file, square_size=50
#     )
# utils.stack_and_resize_images()


# dataset_handler = DatasetHandler(Path("/home/bruno/Desktop/VISIOPE/Predict/"))
# dataset_handler.perform_sanity_check()
# print(dataset_handler.training_couples)
# show_dataset(dataset_handler.training_couples)
# dest_folder = utils.STACKED_PATH
# dataset_handler = DatasetHandler(utils.DATA_PATH)
# train_samples = dataset_handler.samples
# print(train_samples)
# dataset_handler.perform_sanity_check()
# utils.stack_and_resize_images2(
#     dataset_handler.samples, utils.STACKED_PATH, (2000, 2000)
# )
# utils.stack_and_resize_images2(
#     dataset_handler.samples, Path("/home/bruno/Desktop/VISIOPE/Predict/"), (2000, 2000)
# )
# robo = Path(ROBO_PATH + "/train")
mod = GameCartridgeDiscriminator.load_from_checkpoint(
    "/home/bruno/Desktop/VISIOPE/Saves/ckpt/0.0001, 0.0001, 0, 0.01.ckpt"
).to("cpu")
mod.eval()
pred = mod.predict(Path("/home/bruno/Desktop/VISIOPE/Predict/"))
print(pred)
