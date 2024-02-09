from pathlib import Path
from dataset_integrity_checker import DatasetIntegrityChecker
from termcolor import colored
import os
import csv
import cv2
from PIL import Image

import time


class DatasetHandler:
    """Class that implements some methods that perform operations for the dataset"""

    def __init__(self, root_folder: Path, training_folder, valid_folder, test_folder):
        """Initialize the Dataset Handler.

        Args:
            root_folder (Path): The root path where the dataset is stored.
            training_folder (str): The relative path to the training dataset within the root folder.
            valid_folder (str): The relative path to the validation dataset within the root folder.
            test_folder (str): The relative path to the test dataset within the root folder.

        Attributes:
            root_folder (Path): The root path where the dataset is stored.
            training_samples (List[YourDataType]): A list of training samples built from the specified folder.
            eval_samples (List[YourDataType]): A list of validation samples built from the specified folder.
            test_samples (List[YourDataType]): A list of test samples built from the specified folder.

        Note:
            The `_build_data` method is used internally to construct sample lists from the provided folders.

        """
        self.root_folder = root_folder

        # self.training_samples = self._build_data(training_folder)
        # self.eval_samples = self._build_data(valid_folder)
        # self.test_samples = self._build_data(test_folder)
        self.training_samples = self._build_couples()

    def _find_matching_rear_image(self, image_path: str, console):
        """_summary_

        Args:
            image_path (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        couples = []
        front_images = [f for f in os.listdir(image_path) if f.startswith("front")]
        # print(front_images)
        rear_images = [f for f in os.listdir(image_path) if f.startswith("rear")]
        # print(rear_images)
        if len(front_images) != len(rear_images):
            raise ValueError(
                "Number of front and rear images is different in " + str(image_path)
            )
        for f_image in front_images:
            headers = f_image.split("$")
            couples.append(
                (
                    os.path.join(image_path, f_image),
                    os.path.join(
                        image_path, "rear$" + str(headers[1]) + "$" + str(headers[2])
                    ),
                    console,
                )
            )
        return couples

    def _build_couples(self):  # training_folder, eval_folder, test_folder):
        # Implement your logic to build front and rear couples here
        """_summary_

        Args:
            root_folder (str): _description_

        Returns:
            _type_: _description_
        """
        # Get a list of all items (files and subfolders) in the root folder
        res = []
        for console_folder in os.listdir(self.root_folder):
            print(colored(console_folder, "red"))
            for image_folder in os.listdir(self.root_folder / console_folder):
                # print(image_folder)
                res += self._find_matching_rear_image(
                    self.root_folder / console_folder / image_folder, console_folder
                )

        return res

    def perform_sanity_check(self):
        DatasetIntegrityChecker(self.root_folder).dataset_sanity_check()

    def _build_data(self, data_folder):
        image = []
        labels = []
        with open(data_folder / "_classes.csv", "r") as class_csv:
            class_csv.seek(0)
            csv_reader = csv.reader(class_csv)
            # line = next(csv_reader)
            # print(line)

            for file in os.listdir(data_folder):
                # print(file)
                if not file.endswith(".jpg"):
                    continue
                for row in csv_reader:
                    # print(row)
                    if file in row:
                        image.append(cv2.imread(str(data_folder / file)))
                        labels.append(
                            self.return_label(row[1:])
                        )  # discard the string path in pos 0
                        # print(labels)
                        # time.sleep(5)
                        class_csv.seek(0)
                        break

        if len(image) != len(labels):
            raise RuntimeError("Images and labels list lenghts do not match")
        return image, labels

    def return_label(self, row):
        res = []

        for i in range(len(row)):
            if row[i].strip() == "1":
                res.append(i + 1)

        return res
