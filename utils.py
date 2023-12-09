import os
from termcolor import colored
import cv2
from typing import List
import warnings


PATH = os.path.dirname(__file__)
STACKED_PATH = os.path.join(PATH, "Stacked")
DATA_PATH = os.path.join(PATH, "Data")
ROBO_PATH = os.path.join(PATH, "ROBO_DATA")
TRAINING_PATH = os.path.join(PATH, ROBO_PATH + "/train")
VALID_PATH = os.path.join(PATH, ROBO_PATH + "/valid")
TEST_PATH = os.path.join(PATH, ROBO_PATH + "/test")


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


from PIL import Image


def fast_show(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def stack_and_resize_images(image_list, output_path, resize_dimensions=(300, 300)):
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
            print(stacked_image)
            stacked_image.save(
                os.path.join(output_path, str(i) + "$" + image_tuple[0].split("$")[2]),
                "png",
            )
            i += 1
            # print(f"Stacked and resized image saved to: {output_path}")

        except Exception as e:
            print(f"Error: {e}")


# # Example usage:
# image_paths_list = [("image1_path.jpg", "image2_path.jpg")]
# output_path = "output_stacked_image.jpg"
# stack_and_resize_images(image_paths_list[0], output_path)
