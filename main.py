import utils
import model
import random

random.seed(0)
from itertools import product
from gcd_dataset import GameCartridgeDiscriminatorDataset
from gcd_datamodule import GameCartridgeDiscriminatorDatamodule
import tqdm
from model import GameCartridgeDiscriminator
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from termcolor import colored
import os
import subprocess
import threading
import torch
from pytorch_lightning.profilers import PyTorchProfiler
import time
import pytorch_lightning as pl
import gc

# labels_to_idx = {"DS": 1, "GB": 2, "GBA": 3, "true": 4, "false": 5}
# idx_to_labels = {1: "DS", 2: "GB", 3: "GBA", 4: "true", 5: "false"}

labels_to_idx = {
    "DS true": 0,
    "DS false": 1,
    "GBA true": 2,
    "GBA false": 3,
    "GB true": 4,
    "GB false": 5,
}
hyp_comb = list(
    product(
        utils.FC_LR,
        utils.FC_WD,
        utils.FC_DROPOUT,
        utils.CNN_LR,
        utils.CNN_WD,
        utils.NUM_EPOCHS,
    )
)


for hyperparameter in tqdm.tqdm(hyp_comb, colour="yellow", desc="Tried combinations"):
    fc_lr = hyperparameter[0]
    fc_wd = hyperparameter[1]
    fc_dropout = hyperparameter[2]

    cnn_lr = hyperparameter[3]
    cnn_wd = hyperparameter[4]
    num_epochs = hyperparameter[5]

    filename = str(fc_dropout) + ", " + str(cnn_wd) + "model2_with_crop"
    print(filename)
    if filename + ".ckpt" in os.listdir(utils.CKPT_SAVE_DIR_NAME):
        print(colored("CKPT already found, skipping", "red"))
        continue

    print(colored("Built data", "green"))
    gbc_datamodule = GameCartridgeDiscriminatorDatamodule(
        utils.PATH / "train.csv", utils.PATH / "test.csv"
    )

    logger = TensorBoardLogger(save_dir=str(utils.LOG_SAVE_DIR_NAME), name=filename)
    # profiler = PyTorchProfiler(on_trace_ready = torch.profiler.tensorboard_trace_handler("/home/bruno/Desktop/ML_EX2/Saves/tb_logs/profiler0"),trace_memory = True)

    trainer = pl.Trainer(
        log_every_n_steps=20,
        max_epochs=num_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(
                filename=filename,
                monitor="val_loss",
                save_top_k=1,
                every_n_epochs=1,
                mode="min",
                save_weights_only=True,
                verbose=True,
                dirpath=utils.CKPT_SAVE_DIR_NAME,
            ),
        ],
        logger=logger,
        accelerator="gpu",
    )
    print(colored("Built logger and trainer", "green"))
    # labels_name = ["Nothing", "Left", "Right", "Gas", "Brake"]
    gbc_model = GameCartridgeDiscriminator(
        5, None, None, fc_lr, cnn_lr, fc_wd, cnn_wd, fc_dropout, cf_matrix_filename=""
    )

    print(colored("Starting training...", "green"))
    try:
        trainer.fit(gbc_model, datamodule=gbc_datamodule)

    except torch.cuda.OutOfMemoryError as e:
        print((e, colored("Cuda Out of memory detected, skipping", "red")))
        with open("oom.txt ", "a") as f:
            f.write("Cuda" + filename + "\n")
            f.close()
        del gbc_datamodule
        del trainer
        del logger
        del gbc_model
        collected = gc.collect()
        continue

    # print(colored("Starting testing...", "green"))

    # trainer.test(
    #     gbc_model, datamodule=gbc_datamodule
    # )  # ,ckpt_path="best")

    del gbc_datamodule
    del trainer
    del logger
    del gbc_model
    collected = gc.collect()

    # command_thread = threading.Thread(target=subprocess.Popen(['python', "play_policy_template.py"]))


# subprocess.run(["bash", "alert.sh"])
print(colored("Pipeline over", "red"))
