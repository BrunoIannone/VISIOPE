import utils
import model
import random
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

random.seed(utils.RANDOM_SEED)


label_names = ["DS", "GBA", "GB", "true", "false"]


hyp_comb = list(
    product(
        utils.FC_LR,
        utils.FC_WD,
        utils.CNN_LR,
        utils.CNN_WD,
        utils.NUM_EPOCHS,
    )
)


for hyperparameter in tqdm.tqdm(hyp_comb, colour="yellow", desc="Tried combinations"):
    fc_lr = hyperparameter[0]
    fc_wd = hyperparameter[1]

    cnn_lr = hyperparameter[2]
    cnn_wd = hyperparameter[3]
    num_epochs = hyperparameter[4]

    filename = str(fc_lr) + ", " + str(cnn_lr) + ", " + str(fc_wd) + ", " + str(cnn_wd)
    print(filename)

    if filename + ".ckpt" in os.listdir(utils.CKPT_SAVE_DIR_NAME):
        print(colored("CKPT already found, skipping", "red"))
        continue

    print(colored("Built data", "green"))
    gbc_datamodule = GameCartridgeDiscriminatorDatamodule(utils.PATH / "samples.csv")

    logger = TensorBoardLogger(save_dir=str(utils.LOG_SAVE_DIR_NAME), name=filename)

    trainer = pl.Trainer(
        log_every_n_steps=5,
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

    gbc_model = GameCartridgeDiscriminator(
        num_labels=len(label_names),
        label_names=label_names,
        fc_lr=fc_lr,
        cnn_lr=cnn_lr,
        fc_wd=fc_wd,
        cnn_wd=cnn_wd,
        cf_matrix_filename=filename,
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

    print(colored("Starting testing...", "green"))

    trainer.test(
        gbc_model,
        datamodule=gbc_datamodule,
        ckpt_path="best",
    )

    del gbc_datamodule
    del trainer
    del logger
    del gbc_model
    collected = gc.collect()


subprocess.run(["bash", "alert.sh"])
print(colored("Pipeline over", "red"))
