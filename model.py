from typing import List
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import torchmetrics
import utils
from torchmetrics.classification import MultilabelConfusionMatrix
from torchvision.transforms import v2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from dataset_handler import DatasetHandler
import torch.nn as nn


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
    ) -> None:
        """Game cartridge discriminator action model init function

        Args:
            num_labels (int): Number of chosen consoles
            label_names(List[str]): Console names, list of string. Optional
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

        self.resnet_front = utils.CNNF_MODEL
        self.resnet_rear = utils.CNNR_MODEL

        # Remove the last fully connected layer for both models
        self.resnet_front = nn.Sequential(*list(self.resnet_front.children())[:-1])
        self.resnet_rear = nn.Sequential(*list(self.resnet_rear.children())[:-1])

        # Define the classification layers for front and rear images
        self.fc_final = nn.Linear(512 * 2, 5)
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, self.num_labels)

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
        self.conf_mat = MultilabelConfusionMatrix(num_labels=self.num_labels)

        self.save_hyperparameters()

    def forward(self, front_image, rear_image):
        # Forward pass for front image
        features_front = self.resnet_front(front_image)
        features_front = features_front.view(features_front.size(0), -1)

        # Forward pass for rear image
        features_rear = self.resnet_rear(rear_image)
        features_rear = features_rear.view(features_rear.size(0), -1)

        # Concatenate the features from both front and rear images
        combined_features = torch.cat((features_front, features_rear), dim=1)

        # Final classification
        final_output = self.fc_final(combined_features)

        return final_output

    def configure_optimizers(self):
        groups = [
            {
                "params": self.resnet_front.parameters(),
                "lr": self.cnn_lr,
                "weight_decay": self.cnn_wd,
            },
            {
                "params": self.resnet_rear.parameters(),
                "lr": self.cnn_lr,
                "weight_decay": self.cnn_wd,
            },
            {
                "params": self.fc_final.parameters(),
                "lr": self.fc_lr,
                "weight_decay": self.fc_wd,
            },
        ]
        optimizer = torch.optim.AdamW(groups)

        return optimizer

    def training_step(self, train_batch, batch_idx):

        front, rear, labels = train_batch

        outputs = self(front, rear)

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
        front, rear, labels = val_batch

        outputs = self(front, rear)

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
        front, rear, labels = test_batch

        outputs = self(front, rear)
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

    def predict(self, to_predict_path):
        """Predict method

        Args:
            to_predict_path (Path): Prediction folder path

        Returns:
            List: List containing the predicted binary vector for each sample in the folder

        """
        pred = []
        to_predict = self._stack_and_preprocess_for_prediction(to_predict_path)
        for image in to_predict:

            resize_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(256, antialias=True),
                    v2.CenterCrop(224),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            image = resize_transform(image)
            to_predict = image.unsqueeze(0).cpu()

            p = self(to_predict)

            threshold = 0.5
            predicted_probs = torch.sigmoid(p)

            y_pred = (predicted_probs > threshold).float()

            pred.append(y_pred)

        return pred

    def _stack_and_preprocess_for_prediction(self, path):
        to_predict_list = []
        dataset_handler = DatasetHandler(path, True)
        print(dataset_handler.samples)
        i = 0
        for image_tuple in tqdm(dataset_handler.samples, desc="Stacking progress"):
            try:
                # Load images
                image1 = Image.open(image_tuple[0])
                image2 = Image.open(image_tuple[1])

                # Resize images
                width1, height1 = image1.size
                width2, height2 = image2.size

                # Calculate new heights maintaining aspect ratio
                new_height = max(height1, height2)
                new_width1 = int(width1 * (new_height / height1))
                new_width2 = int(width2 * (new_height / height2))

                image1 = image1.resize((new_width1, new_height))
                image2 = image2.resize((new_width2, new_height))

                # Get the size of the stacked image
                new_width = new_width1 + new_width2
                new_height = max(new_height, new_height)

                # Create a new image with the calculated size
                stacked_image = Image.new("RGB", (new_width, new_height))

                # Paste the resized images onto the new image
                stacked_image.paste(image1, (0, 0))
                stacked_image.paste(image2, (new_width1, 0))
                to_predict_list.append(stacked_image)

            except Exception as e:
                print(f"Error: {e}")
        return to_predict_list
