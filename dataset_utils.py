import utils
from PIL import Image
import os
from dataset_handler import DatasetHandler

# utils.manipola_immagine(utils.ROOT_FOOLDER/"Data/test/4",[(None, "_crop")],utils.ROOT_FOOLDER/"Data_crop/test/4")
# train_samples = utils.build_couples(utils.VAL_PATH)
# for file in os.listdir(utils.TEST_DIR_PATH / "0"):
#     utils.draw_random_black_square_on_image(
#         utils.TEST_DIR_PATH / "0" / file, square_size=50
#     )
# utils.stack_and_resize_images()


# dataset_handler = DatasetHandler(utils.PATH, 1, 2, 3)
# dataset_handler.perform_sanity_check()
# print(dataset_handler.training_couples)
# show_dataset(dataset_handler.training_couples)
dest_folder = utils.STACKED_PATH
dataset_handler = DatasetHandler(utils.DATA_PATH)
# train_samples = dataset_handler.samples
# print(train_samples)
# dataset_handler.perform_sanity_check()
utils.stack_and_resize_images2(
    dataset_handler.samples, utils.STACKED_PATH, (2000, 2000)
)
# robo = Path(ROBO_PATH + "/train")
