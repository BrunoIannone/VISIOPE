from termcolor import colored
from utils import DATA_PATH
import os
from pathlib import Path
import re
from typing import List


class DatasetIntegrityChecker:
    def __init__(self, root_folder: Path):
        """DatasetIntegrityChecker init function

        Args:
            root_folder (Path): Root folder Path object. (Path(root_folder_path))
        """
        self.root_folder = root_folder

    def rename_images(self, front_to_translate: str, rear_to_translate: str):
        """Auxiliary function for translating image headers.
        Args:
            front_to_translate (str):Original string to traslate in front
            rear_to_translate (str): Original string to translate in rear

        """

        # Iterate through each folder in the root path
        for console_folder in os.listdir(self.root_folder):
            # Check if the item in the root path is a directory
            if os.path.isdir(self.root_folder / console_folder):
                # Iterate through each console folder in the current folder
                for game_folder in os.listdir(self.root_folder / console_folder):
                    # Check if the item in the console folder is a directory
                    if os.path.isdir(self.root_folder / console_folder / game_folder):
                        for image in os.listdir(
                            self.root_folder / console_folder / game_folder
                        ):
                            # Split the image name using "$"
                            image_parts = image.split("$")

                            # Check if the first part is "fronte" and replace it with "front"
                            if image_parts[0] == front_to_translate:
                                image_parts[0] = "front"
                            # Check if the first part is "retro" and replace it with "rear"
                            elif image_parts[0] == rear_to_translate:
                                image_parts[0] = "rear"

                            # Join the modified parts to get the new image name
                            new_image_name = "$".join(image_parts)

                            # Rename the image file
                            old_image_path = (
                                self.root_folder / console_folder / game_folder / image
                            )
                            new_image_path = (
                                self.root_folder
                                / console_folder
                                / game_folder
                                / new_image_name
                            )
                            os.rename(old_image_path, new_image_path)

        print(colored("Done", "green"))

    def _filename_sanity_check(
        self,
        headers: List[str],
        console_folder: str,
        image_folder: str,
        image: str,
    ):
        """Auxiliary function that performs filenames sanity check. Filename sanity check is successful it every image respect the convention {front,rear}$number${true,false}

        Args:
            headers (List[str]): Splitted image name on "$" symbol (i.e. headers = image_name.split("$")).
            console_folder (str): Path to console folder
            image_folder (str): Path to image folder
            image (str): Path to image folder

        Raises:
            ValueError: If the number of headers is not 3 in the specified image.
            ValueError: If headers[0] is not 'front' or 'rear' in the specified image.
            ValueError: If headers[1] is not a digit in the specified image.
            ValueError: If headers[2] is not 'true' or 'false' in the specified image.

        """
        if len(headers) != 3:
            raise ValueError(
                f"Sanity check failed: in image {image} headers has not lenght 3. \n fault occurred in: "
                + str(self.root_folder / console_folder / image_folder)
            )

        if headers[0] not in ["front", "rear"]:
            raise ValueError(
                f"Sanity check failed: in image {image} the headers[0] not 'front' or 'rear'. \n fault occurred in: "
                + str(self.root_folder / console_folder / image_folder)
            )

        if not headers[1].isdigit():
            raise ValueError(
                "Sanity check failed: headers[1] is not a digit. \n fault occurred in: "
                + str(self.root_folder / console_folder / image_folder)
            )

        label = headers[2].split(".")[0]
        if label not in ["true", "false"]:
            raise ValueError(
                "Sanity check failed: headers[2] is not true or false.\n fault occurred in: "
                + str(self.root_folder / console_folder / image_folder)
            )

    def _search_word(pattern, list, number, label):  # Deprecated
        for elem in list:
            print(elem)
            elem = elem.split("$")
            print(elem[2].split(".")[0])
            if (
                elem[0] == "rear"
                and elem[1] == str(number)
                and elem[2].split(".")[0] == label
            ):
                print(elem)
                return True

        return False

    def _search_word2(
        self, rear_image_set: set, number: int, label: str, extension: str
    ):
        """Auxiliary function that checks whether there exists a the respective rear for a front image.

        Args:
            rear_image_set (set): Set of image names
            number (int): Front number
            label (str): Front label
            extension (str): front extension (jpg,png, etc.)

        Returns:
            Bool: True if there is a match, False otherwise.
        """
        if f"rear${number}${label}.{extension}" in rear_image_set:
            return True

        return False

    def _couples_integrity_check(
        self, front_images: List[str], rear_images: List[str], path: str
    ):
        """Auxiliary function that checks couples integrity. The check is successfull if there is the same number of front and rear images in a folder and
        for each front there is a rear with the same number.

        Args:
            front_images (List[str]): List of front image names
            rear_images (List[str]): List of rear image names
            path (str): Path to images folder

        Raises:
            ValueError: If the number of front and rear images is different in the specified folder ('path').
            ValueError: If any front image does not have a matching rear image in the specified folder ('path').

        """
        if len(front_images) != len(rear_images):
            raise ValueError(
                "Number of front and rear images is different in " + str(path)
            )
        # print(path)
        for image in front_images:
            # print(image)
            pattern = rf'(fronte|retro)\s*\$\s*{re.escape(image.split("$")[1])}\s*(\d+)\s*\$\s*(true|false)\.[a-z]{3}'
            if not self._search_word2(
                rear_images,
                image.split("$")[1],
                image.split("$")[2].split(".")[0],
                image.split("$")[2].split(".")[1],
            ):
                raise ValueError(
                    "Front image not matching with any rear in " + str(path)
                )

    def dataset_sanity_check(self):
        """Function that checks dataset sanity. Check is successfull if image names follow the convention {front,rear}$number${true,false}
        and for each front there is a rear with the same number.

        Args:
            root_folder (str): Root folder path

        Returns:
            bool: True if sanity check is successfull.
        """
        # Get a list of all items (files and subfolders) in the root folder
        print(colored("Beginning dataset sanity check", "yellow"))
        for console_folder in os.listdir(self.root_folder):
            for image_folder in os.listdir(self.root_folder / console_folder):
                for image in os.listdir(
                    self.root_folder / console_folder / image_folder
                ):
                    self._filename_sanity_check(
                        image.split("$"),
                        console_folder=console_folder,
                        image_folder=image_folder,
                        image=image,
                    )

                front_images = set(
                    f
                    for f in os.listdir(
                        self.root_folder / console_folder / image_folder
                    )
                    if f.startswith("front")
                )

                rear_images = set(
                    f
                    for f in os.listdir(
                        self.root_folder / console_folder / image_folder
                    )
                    if f.startswith("rear")
                )
                self._couples_integrity_check(
                    front_images,
                    rear_images,
                    self.root_folder / console_folder / image_folder,
                )
        print(colored("1) Filename sanity check passed", "light_cyan"))
        print(colored("2) Image couples sanity check passed", "light_cyan"))
        print(colored("Dataset sanity check completed successfully", "green"))
        return True


# root_folder = Path(DATA_PATH)

# sanity_checker = DatasetIntegrityChecker(root_folder)
# sanity_checker.rename_images("fronte", "retro")
# sanity_checker.dataset_sanity_check()
