from pathlib import Path
from dataset_integrity_checker import DatasetIntegrityChecker
from termcolor import colored
import os
import time


class DatasetHandler:
    """Class that implements operations for building the dataset"""

    def __init__(self, root_folder: Path, prediction: bool = False):
        """Initialize the Dataset Handler.

        Args:
            root_folder (Path): Dataset path.
            prediction (bool): True if the model has to operate prediction. False otherwise. Default False.

        Attributes:
            root_folder (Path): Dataset path.
            samples (List[tuple]): List containing the sample tuples (front_image_path,rear_image_path,console_name)


        Note:
            The `_build_couples` method is used internally to construct sample lists from the provided folders.
            On the other hand, "_build_couples_for_prediction()" does the same, when prediction = True.
        """
        self.root_folder = root_folder

        if prediction:
            self.samples = self._build_couples_for_prediction()
        else:
            self.samples = self._build_couples()


    def _find_matching_rear_image(self, image_path: str, console: str):
        """Auxiliary function for _build_couples() that finds matching rear image for each front image.

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
        """Builds pairs of front and rear images for all consoles within the root folder.

        Returns:
            list: A list of tuples, each containing paths to matching front and rear images, along with the corresponding console name.
                Each tuple has the following format: (front_image_path,rear_image_path,console_name)

        """

        res = []
        # Explore all console sub-folders to build front/rear couples
        for console_folder in os.listdir(self.root_folder):
            for image_folder in os.listdir(self.root_folder / console_folder):
                res += self._find_matching_rear_image(
                    self.root_folder / console_folder / image_folder, console_folder
                )
        print(colored("Couples build successfully", "green"))
        return res

    def perform_sanity_check(self):  # Dataset integrity checker quick call
        """Performs a sanity check on the dataset by invoking the dataset_sanity_check method of DatasetIntegrityChecker."""
        DatasetIntegrityChecker(self.root_folder).dataset_sanity_check()

    def _build_couples_for_prediction(self):
        """Builds pairs of front and rear images for all consoles and image folders within the root folder. Used for prediction step.

        Returns:
            list: A list of tuples, each containing paths to matching front and rear images, along with the corresponding console object.
                Each tuple has the following format: (front_image_path,rear_image_path,console_name)

        """

        res = []
        # Explore all console sub-folders to build front/rear couples
        for console_folder in os.listdir(self.root_folder):
            # print(colored(console_folder, "red"))
            for image_folder in os.listdir(self.root_folder / console_folder):
                # print(image_folder)
                res += self._find_matching_rear_image_for_prediction(
                    self.root_folder / console_folder / image_folder, console_folder
                )
        print(colored("Couples build successfully", "green"))
        return res

    def _find_matching_rear_image_for_prediction(self, image_path: str):
        """Auxiliary function for _build_couples_for_prediction() that finds matching rear images for each front image.

        Args:
            image_path (str): The path to the directory containing the front and rear images.

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
                        image_path,
                        "rear$" + str(headers[1]),  # str(headers[1]) = number
                    ),  # Rear image path
                )
            )
        return couples
