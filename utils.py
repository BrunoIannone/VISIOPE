import os
from termcolor import colored
PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH,"Data")

def dataset_sanity_check(root_folder):
    # Get a list of all items (files and subfolders) in the root folder
    res = []
    for console_folder in os.listdir(root_folder):
        print(colored(console_folder,"red"))
        for image_folder in os.listdir(os.path.join(root_folder,console_folder)):
            #print(image_folder)
            #res +=  find_matching_rear_image(os.path.join(os.path.join(root_folder,console_folder),image_folder))
            for image in  os.listdir(os.path.join(os.path.join(root_folder,console_folder),image_folder)):
                headers = image.split("$")
                if headers[0] not in ["fronte","retro"]:
                    print("ELLE",os.path.join(root_folder,console_folder))
                    print("sanity check failed" + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
                    raise ValueError("headers[0] not fronte or retro")
                if not headers[1].isdigit():
                    print(colored("sanity check failed","red"))
                    raise ValueError("headers[1] is not a digit")
                label = headers[2].split(".")[0]
                if  label not in ["true","false"]:
                    print("sanity check failed " + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
                    print("ELLE",os.path.join(root_folder,console_folder))
                    print("sanity check failed" + os.path.join(os.path.join(os.path.join(root_folder,console_folder),image_folder),image))
                    
                    raise ValueError("headers[2] is not true or false but is " + str(label))





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