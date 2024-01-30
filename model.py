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
from torchvision import transforms
import numpy as np
import datetime
from transformers import ViTImageProcessor, ViTForImageClassification

# from plot_utils import plot_confusion_matrix


class GameCartridgeDiscriminator(pl.LightningModule):
    def __init__(
        self,
        number_actions: int,
        action_names: List[str] = None,
        action_labels: List[int] = None,
        fc_lr: float = 0.0,
        cnn_lr: float = 0.0,
        fc_wd: float = 0.0,
        cnn_wd: float = 0.0,
        fc_dropout: float = 0.0,
        cf_matrix_filename: str = "",
    ) -> None:
        """Car action model init function

        Args:
            number_actions (int): Number of actions
            action_names(List[str]): Action names list of string. Optional
            action_labels(List[int]): List of action labels in integer values
            fc_lr (float, optional): Linear layer learning rate. Defaults to 0.0.
            cnn_lr (float, optional): CNN learning rate. Defaults to 0.0.
            fc_wd (float, optional): Linear layer weight decay. Defaults to 0.0.
            cnn_wd (float, optional): CNN weight decay. Defaults to 0.0.
            fc_dropout (float, optional): Linear layer dropout . Defaults to 0.0.
            cnn_dropout (float, optional): CNN dropout. Defaults to 0.0.
        """
        super().__init__()
        self.number_actions = number_actions
        self.action_names = action_names
        self.action_labels = action_labels
        self.cf_matrix_filename = cf_matrix_filename
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        self.transformer_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", return_dict=True
        )
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(768, 6)

        self.fc_dropout = nn.Dropout(fc_dropout)

        self.fc_lr = fc_lr
        self.fc_wd = fc_wd
        self.cnn_lr = cnn_lr
        self.cnn_wd = cnn_wd

        self.val_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=number_actions, average="macro"
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=number_actions
        )

        self.test_f1 = torchmetrics.F1Score(
            task="multiclass", num_classes=number_actions, average="macro"
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=number_actions
        )
        self.y_pred = torch.Tensor().cuda().detach()
        self.test_labels = torch.Tensor().cuda().detach()

        self.save_hyperparameters()

    def forward(self, x):
        x = self.transformer_model(**x, output_hidden_states=True)
        # print("ntro")
        # time.sleep(5)
        # Flatten the output
        # x = self.flatten(x)
        x = torch.stack(x.hidden_states[-4:], dim=0).sum(dim=0)
        # print(x.shape)
        # x = self.fc_dropout(x)
        # print(x[1])
        # time.sleep(10)

        # Fully connected layers for classification
        x = self.fc1(x[:, 0, :])
        # x = self.relu(x)
        return x

    def configure_optimizers(self):
        groups = [
            {
                "params": self.fc1.parameters(),
                "lr": 1e-2,
                "weight_decay": 0,
            },
            {
                "params": self.transformer_model.parameters(),
                "lr": 1e-6,
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(groups)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        outputs = self(inputs)

        loss = F.cross_entropy(outputs, labels)

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

        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        outputs = self(inputs)

        y_pred = outputs.argmax(dim=1)
        # print(outputs.shape)
        # print(labels)
        # print(y_pred)
        # time.sleep(10)
        loss = F.cross_entropy(outputs, labels)
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
        y_pred = outputs.argmax(dim=1)
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0).detach()

        self.test_labels = torch.cat((self.test_labels, labels), dim=0).detach()
        loss = F.cross_entropy(outputs, labels)
        self.test_f1(y_pred, labels)
        self.test_accuracy(y_pred, labels)

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
        plot_confusion_matrix(
            self.test_labels.cpu().numpy(),
            self.y_pred.cpu().numpy(),
            "Car action",
            0,
            str(utils.ROOT_FOOLDER) + "/Saves/conf_mat/",
            False,
            True,
            self.action_names,
            self.action_labels,
            cf_matrix_filename=self.cf_matrix_filename,
        )
        del self.y_pred
        del self.test_labels

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
