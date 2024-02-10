import os
from termcolor import colored
import cv2
from typing import List
import warnings
from pathlib import Path
import csv
import time
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime

RANDOM_SEED = 0

############ DIRECTORIES ############
PATH = Path(os.path.dirname(__file__))
STACKED_PATH = PATH / "Stacked Data"
DATA_PATH = PATH / "Data"
LOG_SAVE_DIR_NAME = PATH / "Saves/logs/"
CKPT_SAVE_DIR_NAME = PATH / "Saves/ckpt/"
PLOT_SAVE_PATH = PATH / "Saves/conf_mat/"
#####################################

############ HYPERPARAMETERS ############
NUM_EPOCHS = [1]
BATCH_SIZE = 32
NUM_WORKERS = 4
TEST_SIZE = 0.3
FC_LR = [1e-3]  # , 1e-4, 1e-5]
FC_WD = [0]  # ,0.01,0.1]
CNN_LR = [0]  # , 1e-4]#, 1e-5]
CNN_WD = [0.001]  # ,0.01,0.1]
#########################################


# MODEL_NAME = "google/vit-base-patch16-224"


def show_dataset(image_couples: List[tuple]):
    """Show front and rear image for each sample in image_couples

    Args:
        image_couples (List[tupke]):List containing tuples of string representing front and rear image path. For instance [(front_path.jpg,rear_path.jpg)]
    """
    if image_couples == None or image_couples == []:
        warnings.warn(
            "Invoked show_dataset with image_couples = None or []", RuntimeWarning
        )
        return
    for image in image_couples:
        cv2.namedWindow("Front", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL allows resizing

        # Set the size of the window
        cv2.resizeWindow("Front", 800, 600)

        # print(colored("FRONT","light_cyan"))

        front = cv2.imread(image[0])
        cv2.imshow("Front", front)

        # print(colored("REAR","light_cyan"))

        cv2.namedWindow("Rear", cv2.WINDOW_NORMAL)  # cv2.WINDOW_NORMAL allows resizing
        cv2.resizeWindow("Rear", 800, 600)
        rear = cv2.imread(image[1])

        cv2.imshow("Rear", rear)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            print(colored("Exiting...", "red"))
            return

    print(colored("Showed last image", "green"))


def fast_show(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stack_and_resize_images(
    image_list, output_path, resize_dimensions=(300, 300)
):  # deprecated
    i = 0
    for image_tuple in image_list:
        # print([image_tuple])
        try:
            # Load images
            image1 = Image.open(image_tuple[0])
            image2 = Image.open(image_tuple[1])

            # Resize images
            image1 = image1.resize(resize_dimensions)
            image2 = image2.resize(resize_dimensions)

            # Get the size of the stacked image
            new_width = image1.width + image2.width
            new_height = max(image1.height, image2.height)

            # Create a new image with the calculated size
            stacked_image = Image.new("RGB", (new_width, new_height))

            # Paste the resized images onto the new image
            stacked_image.paste(image1, (0, 0))
            stacked_image.paste(image2, (image1.width, 0))

            # Save the stacked and resized image to the specified output path
            # print(stacked_image)
            # stacked_image.save(
            #     os.path.join(output_path, str(i) + "$" + image_tuple[0].split("$")[2]),
            #     "png",
            # )
            print(
                os.path.join(output_path, str(i) + "$" + image_tuple[0].split("$")[2])
            )

            # time.sleep(10)
            stacked_image.save(
                os.path.join(
                    output_path
                    / str(
                        image_tuple[2]
                        + " "
                        + image_tuple[0].split("$")[2].split(".")[0]
                    ),
                    str(i) + "$" + image_tuple[0].split("$")[2],
                ),
            )
            i += 1
            # print(f"Stacked and resized image saved to: {output_path}")

        except Exception as e:
            print(f"Error: {e}")


def stack_and_resize_images2(image_list, output_path, resize_dimensions=(300, 300)):
    i = 0
    for image_tuple in tqdm(image_list, desc="Stacking progress"):
        try:
            # Load images
            image1 = Image.open(image_tuple[0])
            image2 = Image.open(image_tuple[1])

            # Resize images
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

            # Paste the resized images onto the new image
            stacked_image.paste(image1, (0, 0))
            stacked_image.paste(image2, (new_width1, 0))

            # Save the stacked and resized image to the specified output path
            stacked_image.save(
                os.path.join(
                    output_path
                    / str(
                        image_tuple[2]
                        + " "
                        + image_tuple[0].split("$")[2].split(".")[0]
                    ),
                    str(i) + "$" + image_tuple[0].split("$")[2],
                ),
            )
            i += 1

        except Exception as e:
            print(f"Error: {e}")


def build_couples(dir):
    """Build a csv having rows with the following structure: sample_path, label (taken from folder name)

    Args:
        dir (str): Dataset folder


    """
    res = []
    for console_folder in os.listdir(dir):
        for cart_image in os.listdir(dir / console_folder):
            res.append((dir / console_folder / cart_image, console_folder))

    csv_file_path = "couples.csv"
    # Open the CSV file in write mode
    with open(csv_file_path, "w", newline="") as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["Image", "Label"])
        for elem in res:
            # Write the predictions to the CSV file
            csv_writer.writerow((str(elem[0]), elem[1]))

    print(f"Train data been written to {csv_file_path}.")


def save_conf_mat(fig_, cf_matrix_filename):
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
