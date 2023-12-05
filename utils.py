import os
from termcolor import colored
PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH,"Data")



          

#TODO: Improve this function accuracy and visualization quality  


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
        
        for r_image in rear_images:
            headers2 = r_image.split("$")
            if headers[1] == headers[2]:
                couples.append((f_image,r_image))
        couples.append((os.path.join(image_path,f_image),os.path.join(image_path,"retro$"+str(headers[1]) + "$"+str(headers[2]))))
    return couples