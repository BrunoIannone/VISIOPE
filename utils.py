import os
from termcolor import colored
import cv2
from typing import List
import warnings

PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH, "Data")


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
