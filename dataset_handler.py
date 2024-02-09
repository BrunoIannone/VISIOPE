from pathlib import Path
from dataset_integrity_checker import DatasetIntegrityChecker
from termcolor import colored
import os
import csv
import cv2
from PIL import Image

import time


class DatasetHandler:
    """Class that implements some operations for the dataset"""

    def __init__(self, root_folder: Path):
        """Initialize the Dataset Handler.

        Args:
            root_folder (Path): Dataset path.


        Attributes:
            root_folder (Path): Dataset path.
            samples (List[tuple]): List containing the sample tuples (front_image_path,rear_image_path,console_name)


        Note:
            The `_build_couples` method is used internally to construct sample lists from the provided folders.

        """
        self.root_folder = root_folder

        self.samples = self._build_couples()

    def _find_matching_rear_image(self, image_path: str, console):
        """Auxiliary finction for _build_couples() that finds matching rear images for each front image.

        Args:
            image_path (str): The path to the directory containing the front and rear images.
            console (str): The cartridge belonging console.

        Raises:
            ValueError: If the number of front and rear images is different in the specified directory.

        Returns:
            list: A list of tuples, each containing paths to matching front and rear images, along with the belonging console. (front_image_path,rear_image_path,console_name)

        """
        couples = []
        # Take front images
        front_images = [f for f in os.listdir(image_path) if f.startswith("front")]

        # Take rear images
        rear_images = [f for f in os.listdir(image_path) if f.startswith("rear")]

        if len(front_images) != len(
            rear_images
        ):  # Mismatching number of front and rear images
            raise ValueError(
                "Number of front and rear images is different in " + str(image_path)
            )
        for f_image in front_images:  # Tuple building
            headers = f_image.split("$")
            couples.append(
                (
                    os.path.join(image_path, f_image),  # Front image path
                    os.path.join(
                        image_path, "rear$" + str(headers[1]) + "$" + str(headers[2])
                    ),  # Rear image path
                    console,  # Belonging console
                )
            )
        return couples

    def _build_couples(self):
        """Builds pairs of front and rear images for all consoles and image folders within the root folder.

        Returns:
            list: A list of tuples, each containing paths to matching front and rear images, along with the corresponding console object.
                Each tuple has the following format:
                - (str, str, console)
                    - The full path to the front image.
                    - The full path to the corresponding rear image.
                    - The console object for logging or output.
        """

        res = []
        # Explore all console sub-folders to build front/rear couples
        for console_folder in os.listdir(self.root_folder):
            # print(colored(console_folder, "red"))
            for image_folder in os.listdir(self.root_folder / console_folder):
                # print(image_folder)
                res += self._find_matching_rear_image(
                    self.root_folder / console_folder / image_folder, console_folder
                )
        print(colored("Couples build successfully", "green"))
        return res

    def perform_sanity_check(self):  # Dataset integrity checker quick call
        """Performs a sanity check on the dataset by invoking the dataset_sanity_check method of DatasetIntegrityChecker.

        This method is responsible for initiating a dataset integrity check on the specified root folder. It creates an instance of DatasetIntegrityChecker with the root folder and invokes its dataset_sanity_check method to perform the actual sanity check.


        """
        DatasetIntegrityChecker(self.root_folder).dataset_sanity_check()

    # def _build_data(self, data_folder):
    #     image = []
    #     labels = []
    #     with open(data_folder / "_classes.csv", "r") as class_csv:
    #         class_csv.seek(0)
    #         csv_reader = csv.reader(class_csv)
    #         # line = next(csv_reader)
    #         # print(line)

    #         for file in os.listdir(data_folder):
    #             # print(file)
    #             if not file.endswith(".jpg"):
    #                 continue
    #             for row in csv_reader:
    #                 # print(row)
    #                 if file in row:
    #                     image.append(cv2.imread(str(data_folder / file)))
    #                     labels.append(
    #                         self.return_label(row[1:])
    #                     )  # discard the string path in pos 0
    #                     # print(labels)
    #                     # time.sleep(5)
    #                     class_csv.seek(0)
    #                     break

    #     if len(image) != len(labels):
    #         raise RuntimeError("Images and labels list lenghts do not match")
    #     return image, labels

    # def return_label(self, row):
    #     res = []

    #     for i in range(len(row)):
    #         if row[i].strip() == "1":
    #             res.append(i + 1)

    #     return res
