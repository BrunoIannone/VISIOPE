import utils
from PIL import Image
import os

# utils.manipola_immagine(utils.ROOT_FOOLDER/"Data/test/4",[(None, "_crop")],utils.ROOT_FOOLDER/"Data_crop/test/4")
train_samples = utils.build_couples(utils.TEST_PATH)
# for file in os.listdir(utils.TEST_DIR_PATH / "0"):
#     utils.draw_random_black_square_on_image(
#         utils.TEST_DIR_PATH / "0" / file, square_size=50
#     )
