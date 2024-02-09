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
        num_labels: int,
        console_names: List[str] = None,
        console_labels: List[int] = None,
        fc_lr: float = 0.0,
        cnn_lr: float = 0.0,
        fc_wd: float = 0.0,
        cnn_wd: float = 0.0,
        fc_dropout: float = 0.0,
        cf_matrix_filename: str = "",
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
        self.console_names = console_names
        self.console_labels = console_labels
        self.cf_matrix_filename = cf_matrix_filename
        self.processor = ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=5,
            ignore_mismatched_sizes=True,
        )
        self.transformer_model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            return_dict=True,
            num_labels=5,
            ignore_mismatched_sizes=True,
        )
        self.flatten = nn.Flatten()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(768, 5)

        self.fc_dropout = nn.Dropout(fc_dropout)

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
        self.y_pred = torch.Tensor().cuda().detach()
        self.test_labels = torch.Tensor().cuda().detach()

        self.save_hyperparameters()

    def forward(self, x):
        x = self.transformer_model(**x, output_hidden_states=True)
        hidden_states = x.hidden_states[-1]  # Access the last layer's hidden states
        cls_token = hidden_states[:, 0, :]  # Select the CLS token

        # Apply dropout if needed
        cls_token = self.fc_dropout(cls_token)

        # Pass through the linear layer
        logits = self.fc1(cls_token)
        # print(logits)
        return logits

    def configure_optimizers(self):
        groups = [
            {
                "params": self.fc1.parameters(),
                "lr": 1e-2,
                "weight_decay": 0,
            },
            {
                "params": self.transformer_model.parameters(),
                "lr": 1e-5,
                "weight_decay": 0,
            },
        ]
        optimizer = torch.optim.AdamW(groups)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        # time.sleep(5)
        image, labels = train_batch
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs["pixel_values"] = inputs["pixel_values"].cuda()
        # print(inputs["pixel_values"].shape)
        outputs = self(inputs)
        # print(outputs)
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

        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        outputs = self(inputs)

        threshold = 0.5
        predicted_probs = torch.sigmoid(outputs)
        # print(predicted_probs)
        y_pred = (predicted_probs > threshold).float()
        # print(y_pred)
        # time.sleep(10)
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
        # print(image.shape)
        # time.sleep(100)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        outputs = self(inputs)
        threshold = 0.5
        predicted_probs = torch.sigmoid(outputs)
        y_pred = (predicted_probs > threshold).float()
        self.y_pred = torch.cat((self.y_pred, y_pred), dim=0).detach()

        self.test_labels = torch.cat((self.test_labels, labels), dim=0).detach()
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
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

    # def on_test_end(self) -> None:
    #     plot_confusion_matrix(
    #         self.test_labels.cpu().numpy(),
    #         self.y_pred.cpu().numpy(),
    #         "Car action",
    #         0,
    #         str(utils.ROOT_FOOLDER) + "/Saves/conf_mat/",
    #         False,
    #         True,
    #         self.action_names,
    #         self.action_labels,
    #         cf_matrix_filename=self.cf_matrix_filename,
    #     )
    #     del self.y_pred
    #     del self.test_labels

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
