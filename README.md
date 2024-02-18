# VISIOPE

## Project Overview

This is the project for *"Vision and perception"* exam.

## Installing dependencies

Before running the project it is necessary to install all dependencies. Do it by running 

```pip install -r requirements.txt```

## Files

1. main.py: 

    This is the main script for executing the training process.

2. utils.py

    In this file, you'll declare the model hyperparameters. Moreover, it contains some auxiliary functions.
    
3. model.py

    This file contains the  *"GameCartridgeDiscriminator"* model.

4. dataset_handler.py

    Implement the  *"DatasetHandler"* class that contains methods for building the dataset.

5. dataset_integrity_checker.py

   Implement the class *"DatasetIntegrityCheckerChecks"* that checks if the dataset follows the correct naming conventions and that there are fronts for each rear image.

6. data_processor.py

    Implement the *"DataProcessor"* class that builds training, validation and test set.

7. gcd_dataset.py and gcd_datamodule.py

    These two files contains respectively the *"GameCartridgeDiscriminatorDataset"* class and *"GameCartridgeDiscriminatorDatamodule"* class. The former creates the couples (sample, label) while the latter loads these during the actual training, validation or test step.

8. load_data.py

    Reads the .csv file containing the samples information (a .csv can be build whith the build_couples() method in utils.py)


## Folders structure and usage

1. Data

    The data folder expect to have a folder for each console and, in each console folder, there are sub-folders containing the images.
    For instance:

    -**Data**

        *DS
            ** Foto_folder_1
                *** front$1$true.png
                *** rear$1$true.png
                *** front$2$false.png
                *** rear$2$false.png
        *GB

        *GBA


    Notice the image names that follow the following conventions **{front,rear}$number${true,false}.extension** where number is a unique numeric identifier for each couple. You can use the *DatasetIntegrityChecker* class to perform the integrity check, or the quick call in the *DatasetHandler* class.

2. Stacked Data

    Once the dataset is ready,  it is possible to use the *stack_front_rear_images()* method in utils.py to create the Stacked Data folder. These images will be the actual input to the model

3. Predict

    The prediction folder follows the same structure of Data folder, but images names won't have the {true,false} part. The model *predict()* method will take care of stacking the images, so no need on stacking them before. 

4. Saves

    This folder contains 3 subfolders: ckpt, conf_mat and logs. In the first one the model checkpoints (.ckpt) are stored; in the second one the confusion matrix on test end is saved as .png while on the last one are saved the logs of the model evolution during training (logging is performed with Tensorboard)


## In conclusion

    Once the Stacked Data folder is ready, the training pipeline can be launched by executing the main.py. An example of building the dataset and predicting new samples can be found in dataset_utils.py.
    
**N.B. The folder structure is not included in the repo and must be replicated accordingly**
    