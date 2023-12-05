from termcolor import colored
import os
def filename_sanity_check(headers,root_folder,console_folder,image_folder,image):

    if headers[0] not in ["fronte","retro"]:
                    raise ValueError("Sanity check failed: headers[0] not fronte or retro. \n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
                
    if not headers[1].isdigit():
        raise ValueError("Sanity check failed: headers[1] is not a digit. \n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
    
    label = headers[2].split(".")[0]
    if  label not in ["true","false"]:
        raise ValueError("Sanity check failed: headers[2] is not true or false.\n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))


def dataset_sanity_check(root_folder):
    # Get a list of all items (files and subfolders) in the root folder
    res = []


    for console_folder in os.listdir(root_folder):
        print(colored(console_folder,"red"))
        for image_folder in os.listdir(os.path.join(root_folder,console_folder)):
            front_images = [f for f in os.listdir(os.path.join(os.path.join(root_folder,console_folder),image_folder)) if f.startswith('fronte')]
            #print(front_images)
            rear_images = [f for f in os.listdir(os.path.join(os.path.join(root_folder,console_folder),image_folder)) if f.startswith('retro')]
            #print(rear_images)
            if len(front_images) != len(rear_images):
                raise ValueError("Number of front and rear images is different in " + str(image_folder))
            for image in  os.listdir(os.path.join(os.path.join(root_folder,console_folder),image_folder)):
                filename_sanity_check(image.split("$"),root_folder=root_folder,console_folder=console_folder,image_folder=image_folder,image=image)
