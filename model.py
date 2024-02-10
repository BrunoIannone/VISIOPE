from typing import List
from pytorch_lightning.utilities.types import (
    EVAL_DATALOADERS,
    STEP_OUTPUT,
    TRAIN_DATALOADERS,
)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils
from torchmetrics.classification import MultilabelConfusionMatrix
from torchvision import transforms, models
import numpy as np
import datetime
from transformers import ViTImageProcessor, ViTForImageClassification
import os

# from plot_utils import plot_confusion_matrix


class GameCartridgeDiscriminator(pl.LightningModule):
    def __init__(
        self,
        num_labels: int,
        label_names: List[str] = None,
        fc_lr: float = 0.0,
        cnn_lr: float = 0.0,
        fc_wd: float = 0.0,
        cnn_wd: float = 0.0,
        cf_matrix_filename: str = "",
        plot_save_path: str = "",
    ) -> None:
        """Car action model init function

        Args:
            num_labels (int): Number of chosen consoles
            console_names(List[str]): Console names, list of string. Optional
            console_labels(List[int]): List of console labels in integer values
            fc_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
            cnn_lr (float, optional): CNN learning rate. Defaults to 0.0.
            fc_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
            cnn_wd (float, optional): CNN weight decay. Defaults to 0.0.
            fc_dropout (float, optional): Linear layer dropout . Defaults to 0.0.
            cnn_dropout (float, optional): CNN dropout. Defaults to 0.0.
        """
        super().__init__()
        self.num_labels = num_labels
        self.label_names = label_names

        self.cf_matrix_filename = cf_matrix_filename

        self.model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 5)

        self.fc_lr = fc_lr
        self.fc_wd = fc_wd
        self.cnn_lr = cnn_lr
        self.cnn_wd = cnn_wd

        self.val_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multilabel", num_labels=self.num_labels
        )

        self.test_f1 = torchmetrics.F1Score(
            task="multilabel", num_labels=self.num_labels, average="macro"
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multilabel", num_labels=self.num_labels
        )
        self.conf_mat = MultilabelConfusionMatrix(num_labels=5)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        groups = [
            {
                "params": self.model.fc.parameters(),
                "lr": self.fc_lr,
                "weight_decay": self.fc_wd,
            },
            {
                "params": self.model.parameters(),
                "lr": self.cnn_lr,
                "weight_decay": self.cnn_wd,
            },
        ]
        optimizer = torch.optim.AdamW(self.model.parameters())
        return optimizer

    def training_step(self, train_batch, batch_idx):

        image, labels = train_batch

        outputs = self(image)

        loss = F.binary_cross_entropy_with_logits(outputs, labels)

        self.log_dict(
            {"train_loss": loss},
            on_epoch=True,
            batch_size=utils.BATCH_SIZE,
            on_step=False,
            prog_bar=True,
            enable_graph=False,
        )
        return loss

    def validation_step(self, val_batch, idx):
        image, labels = val_batch

        outputs = self(image)

        threshold = 0.5
        predicted_probs = torch.sigmoid(outputs)

        y_pred = (predicted_probs > threshold).float()

        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.val_f1(y_pred, labels)
        self.val_accuracy(y_pred, labels)
        self.log_dict(
            {"val_loss": loss, "valid_f1": self.val_f1, "valid_acc": self.val_accuracy},
            batch_size=utils.BATCH_SIZE,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            enable_graph=False,
        )

    def test_step(self, test_batch):
        image, labels = test_batch

        outputs = self(image)
        threshold = 0.5
        predicted_probs = torch.sigmoid(outputs)
        y_pred = (predicted_probs > threshold).float()

        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        self.test_f1(y_pred, labels)
        self.test_accuracy(y_pred, labels)

        self.conf_mat.update(y_pred.to(torch.int64), labels.to(torch.int64))
        self.log_dict(
            {
                "test_loss": loss,
                "test_f1": self.test_f1,
                "test_acc": self.test_accuracy,
            },
            batch_size=utils.BATCH_SIZE,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            enable_graph=False,
        )

    def on_test_end(self) -> None:

        fig_, ax_ = self.conf_mat.plot(labels=self.label_names)
        utils.save_conf_mat(fig_, self.cf_matrix_filename)

    def predict(self, to_predict):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((96, 96)),
                transforms.ToTensor(),
            ]
        )
        to_predict = transform(to_predict).unsqueeze(0).cpu()

        p = self(to_predict)

        _, action = torch.max(p, 1)
        return int(action)
