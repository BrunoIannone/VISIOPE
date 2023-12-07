from termcolor import colored
from utils import DATA_PATH
import os
from pathlib import Path
import re
def rename_images(root_path):
    # Iterate through each folder in the root path
    for console_folder in os.listdir(root_path):
    
        # Check if the item in the root path is a directory
        if os.path.isdir(root_path/console_folder):
            
            # Iterate through each console folder in the current folder
            for game_folder in os.listdir(root_path/console_folder):
                
                # Check if the item in the console folder is a directory
                if os.path.isdir(root_path/console_folder/game_folder):
                    
                    for image in os.listdir(root_path/console_folder/game_folder):
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
        
                        # Rename the image file
                        old_image_path = root_path/console_folder/game_folder/image
                        new_image_path = root_path/console_folder/game_folder/new_image_name
                        os.rename(old_image_path, new_image_path)

    print("Done")
     
def filename_sanity_check(headers,root_folder,console_folder,image_folder,image):

    if headers[0] not in ["front","rear"]:
                    raise ValueError(f"Sanity check failed: in image {image} the headers[0] not 'front' or 'rear'. \n fault occurred in: " + str(root_folder/console_folder/image_folder))
                
    if not headers[1].isdigit():
        raise ValueError("Sanity check failed: headers[1] is not a digit. \n fault occurred in: " + str(root_folder/console_folder/image_folder))
    
    label = headers[2].split(".")[0]
    if  label not in ["true","false"]:
        raise ValueError("Sanity check failed: headers[2] is not true or false.\n fault occurred in: " + str(root_folder/console_folder/image_folder))


def search_word(pattern,list,number,label):
    #print("CI STO")
    #print(list)
    for elem in list:
        print(elem)
        elem = elem.split("$")
        print(elem[2].split(".")[0])
        if elem[0] == 'rear' and elem[1] == str(number) and elem[2].split(".")[0] == label:
            print(elem)
            return True
        
    return False

def search_word2(pattern,rear_image_set,number,label,extension):
    if f'rear${number}${label}.{extension}' in rear_image_set:
        return True
    
    return False
def couples_integrity_check(front_images,rear_images,path):
    #print("CI STO")

    if len(front_images) != len(rear_images):
                raise ValueError("Number of front and rear images is different in " + str(path))
    #print(path)        
    for image in front_images:
        #print(image)
        pattern = rf'(fronte|retro)\s*\$\s*{re.escape(image.split("$")[1])}\s*(\d+)\s*\$\s*(true|false)\.[a-z]{3}'
        if not search_word2(pattern,rear_images,image.split("$")[1],image.split("$")[2].split(".")[0],image.split("$")[2].split(".")[1]):
          
            raise ValueError ("Front image not matching with any rear in " + str(path))   
def dataset_sanity_check(root_folder):
    # Get a list of all items (files and subfolders) in the root folder
    print(colored("Beginning dataset sanity check","yellow"))
    for console_folder in os.listdir(root_folder):
        
        for image_folder in os.listdir(root_folder/console_folder):
            for image in  os.listdir(root_folder/console_folder/image_folder):
                filename_sanity_check(image.split("$"),root_folder=root_folder,console_folder=console_folder,image_folder=image_folder,image=image)
            
             
            front_images = set(f for f in os.listdir(root_folder/console_folder/image_folder) if f.startswith('front'))
            
            rear_images = set(f for f in os.listdir(root_folder/console_folder/image_folder) if f.startswith('rear'))
            couples_integrity_check(front_images,rear_images,root_folder/console_folder/image_folder)
    print(colored("1) Filename sanity check passed","light_cyan"))
    print(colored("2) Image couples sanity check passed","light_cyan"))    
    print(colored("Dataset sanity check completed successfully","green"))        


root_folder = Path(DATA_PATH)
#rename_images(root_folder)
dataset_sanity_check(root_folder)