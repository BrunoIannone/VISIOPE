import os
from typing import List
from pathlib import Path
import csv
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
from torchvision import models


RANDOM_SEED = 0

############ DIRECTORIES ############
PATH = Path(os.path.dirname(__file__))
STACKED_PATH = PATH / "Stacked Data"
DATA_PATH = PATH / "Data"
LOG_SAVE_DIR_NAME = PATH / "Saves/logs/"
CKPT_SAVE_DIR_NAME = PATH / "Saves/ckpt"
PLOT_SAVE_PATH = PATH / "Saves/conf_mat/"
PREDICT_PATH = PATH / "Predict"
#####################################

############ HYPERPARAMETERS ############
NUM_EPOCHS = [100]
BATCH_SIZE = 128
NUM_WORKERS = 8
TEST_SIZE = 0.3
FC_LR = [1e-3, 1e-4]
FC_WD = [0, 0.001, 0.01, 0.1]
CNN_LR = [1e-4, 1e-5]
CNN_WD = [0, 0.001, 0.01, 0.1]
CNNF_MODEL = models.resnet18(weights="ResNet18_Weights.DEFAULT")  # Change on needs
CNNR_MODEL = models.resnet18(weights="ResNet18_Weights.DEFAULT")  # Change on needs

#########################################


def stack_front_rear_images(image_list: List, output_path: str):
    """Stack front and rear images.

    Args:
        image_list (List): List containing tuples with the structure (front_img_path,rear_img_path,console_name).
        N.B.: front and rear image names follow the following structure: {front,rear}$number${true,false}.<extension>
        output_path (str): Output path for saving.
    """

    i = 0  # Index for unique naming
    for image_tuple in tqdm(image_list, desc="Stacking progress"):
        try:
            # Load images
            image1 = Image.open(image_tuple[0])
            image2 = Image.open(image_tuple[1])

            # Image sizes
            width1, height1 = image1.size
            width2, height2 = image2.size

            # Calculate new heights maintaining aspect ratio
            new_height = max(height1, height2)
            new_width1 = int(width1 * (new_height / height1))
            new_width2 = int(width2 * (new_height / height2))

            image1 = image1.resize((new_width1, new_height))
            image2 = image2.resize((new_width2, new_height))

            # Get the size of the stacked image
            new_width = new_width1 + new_width2
            new_height = max(new_height, new_height)

            # Create a new image with the calculated size
            stacked_image = Image.new("RGB", (new_width, new_height))

            # Paste the images onto the new image
            stacked_image.paste(image1, (0, 0))
            stacked_image.paste(image2, (new_width1, 0))

            # Save the stacked images to the specified output path
            stacked_image.save(
                os.path.join(
                    output_path
                    / str(
                        image_tuple[2]  # console
                        + " "
                        + image_tuple[0].split("$")[2].split(".")[0]  # {true,false}
                    ),
                    str(i)
                    + "$"
                    + image_tuple[0].split("$")[2],  # i${true,false}.<extension>
                ),
            )
            i += 1

        except Exception as e:
            print(f"Error: {e}")


def build_couples(samples: List):
    """Build a .csv having rows with the following structure: sample_path, console {true,false}

    Args:
        dir (Path): Path to dataset folder


    """
    res = []
    print(samples, "PENE")
    csv_file_path = "couples.csv"
    # Open the .CSV file in write mode
    with open(csv_file_path, "w", newline="") as csv_file:
        # Create a .CSV writer object
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Front£Rear", "Label"])
        for elem in samples:
            label = elem[0].split("$")[2].split(".")[0]
            print(label)
            # Write the predictions to the .CSV file
            csv_writer.writerow((str(elem[0] + "£" + elem[1]), elem[2] + " " + label))

    print(f"Train data been written to {csv_file_path}.")


def save_conf_mat(fig_, cf_matrix_filename: str):
    """Confusion matrix save function

    Args:
        fig_ (Fig Obj): A figure object
        cf_matrix_filename (str): Confusion matrix filename
    """
    fig_.set_size_inches(1920 / 100, 1080 / 100)
    if cf_matrix_filename != "":
        plt.savefig(
            os.path.join(
                PLOT_SAVE_PATH,
                "confusion_matrix"
                + cf_matrix_filename
                + str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
                + ".png",
            ),
            dpi=200,
        )

        plt.clf()
        plt.close()
