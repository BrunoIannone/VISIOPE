import os
from utils import DATA_PATH, dataset_sanity_check
from termcolor import colored
from PIL import Image

def find_matching_rear_image(image_path):
    couples = []
    front_images = [f for f in os.listdir(image_path) if f.startswith('fronte')]
    #print(front_images)
    rear_images = [f for f in os.listdir(image_path) if f.startswith('retro')]
    #print(rear_images)
    if len(front_images) != len(rear_images):
        raise ValueError("Number of front and rear images is different in " + str(image_path))
    for f_image in front_images:
        headers = f_image.split("$") 
        couples.append((os.path.join(image_path,f_image),os.path.join(image_path,"retro$"+str(headers[1]) + "$"+str(headers[2]))))
    return couples

def build_couples(root_folder):
    # Get a list of all items (files and subfolders) in the root folder
    res = []
    for console_folder in os.listdir(root_folder):
        print(colored(console_folder,"red"))
        for image_folder in os.listdir(os.path.join(root_folder,console_folder)):
            #print(image_folder)
            res +=  find_matching_rear_image(os.path.join(os.path.join(root_folder,console_folder),image_folder))
                
    return res
    
root_folder = DATA_PATH
#couples = build_couples(root_folder)
#print(couples)
dataset_sanity_check(DATA_PATH)
# for image in couples:
#     Image.open(image[0]).show()
    
#     Image.open(image[1]).show()
    
    