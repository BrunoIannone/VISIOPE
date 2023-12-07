from termcolor import colored
from utils import DATA_PATH
import os
from pathlib import Path
import re
def rename_images(root_path):
    # Iterate through each folder in the root path
    for folder in os.listdir(root_path):
        print("rooth_path",root_path)
        folder_path = os.path.join(root_path, folder)

        # Check if the item in the root path is a directory
        if os.path.isdir(folder_path):
            print("console_folder",folder_path)
            # Iterate through each console folder in the current folder
            for console_folder in os.listdir(folder_path):
                console_path = os.path.join(folder_path, console_folder)

                # Check if the item in the console folder is a directory
                if os.path.isdir(console_path):
                    print("game_path",console_path)
                    # Iterate through each image folder in the console folder
                    for image in os.listdir(console_path):
                        print("image",image)
                        image_path = os.path.join(console_path, image)

                        
                        

                        # Check if the item in the image folder is a file
                        
                        print(colored("ENTRp","green"))
                        # Split the image name using "$"
                        image_parts = image.split("$")

                        # Check if the first part is "fronte" and replace it with "front"
                        if image_parts[0] == "fronte":
                            image_parts[0] = "front"
                        # Check if the first part is "retro" and replace it with "rear"
                        elif image_parts[0] == "retro":
                            image_parts[0] = "rear"

                        # Join the modified parts to get the new image name
                        new_image_name = "$".join(image_parts)
                        print(colored("IMAGE PARTS","red"),image_parts)
                        # Rename the image file
                        new_image_path = os.path.join(console_path, new_image_name)
                        os.rename(image_path, new_image_path)

    print("Done")
     
def filename_sanity_check(headers,root_folder,console_folder,image_folder,image):

    if headers[0] not in ["fronte","retro"]:
                    raise ValueError("Sanity check failed: headers[0] not fronte or retro. \n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
                
    if not headers[1].isdigit():
        raise ValueError("Sanity check failed: headers[1] is not a digit. \n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
    
    label = headers[2].split(".")[0]
    if  label not in ["true","false"]:
        raise ValueError("Sanity check failed: headers[2] is not true or false.\n fault occurred in: " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))


def search_word(pattern,list,number,label):
    print(list)
    for elem in list:
        print(elem)
        elem = elem.split("$")
        print(elem[2].split(".")[0])
        if elem[0] == 'retro' and elem[1] == str(number) and elem[2].split(".")[0] == label:
            print(elem)
            return True
        
    return False
def couples_integrity_check(front_images,rear_images,path):
     
     for image in front_images:
          pattern = rf'(fronte|retro)\s*\$\s*{re.escape(image.split("$")[1])}\s*(\d+)\s*\$\s*(true|false)\.\w+'
          if not search_word(pattern,rear_images,image.split("$")[1],image.split("$")[2].split(".")[0]):
          
            raise ValueError ("Front image not matching with any rear in " + str(path))   
def dataset_sanity_check(root_folder):
    # Get a list of all items (files and subfolders) in the root folder
    res = []


    for console_folder in os.listdir(root_folder):
        print(colored(console_folder,"red"))
        for image_folder in os.listdir(root_folder/console_folder):
            for image in  os.listdir(root_folder/console_folder/image_folder):
                filename_sanity_check(image.split("$"),root_folder=root_folder,console_folder=console_folder,image_folder=image_folder,image=image)

            front_images = [f for f in os.listdir(root_folder/console_folder/image_folder) if f.startswith('fronte')]
            #print(front_images)
            rear_images = [f for f in os.listdir(root_folder/console_folder/image_folder) if f.startswith('retro')]
            couples_integrity_check(front_images,rear_images,root_folder/console_folder/image_folder)
            
            #print(rear_images)
            if len(front_images) != len(rear_images):
                raise ValueError("Number of front and rear images is different in " + root_folder/console_folder/image_folder)
            


root_folder = Path(DATA_PATH)
#rename_images(root_folder)
#dataset_sanity_check(root_folder)