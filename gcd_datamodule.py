from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import gcd_dataset
import utils
from data_processor import DataProcessor
import torch
from pathlib import Path


class GameCartridgeDiscriminatorDatamodule(LightningDataModule):
    """Datamodule for game cartridge discriminator dataset."""

    def __init__(self, samples_path: Path):
        """Init function for game cartridge discriminator datamodule

        Args:
            samples_path (Path): Path to samples csv containing rows with the following structure: sample_path, console {true,false}
        """
        super().__init__()

        self.samples_path = samples_path

    def setup(self, stage: str):
        data_processor = DataProcessor(
            self.samples_path, utils.TEST_SIZE, utils.RANDOM_SEED
        )
        if stage == "fit":
            self.train_dataset = gcd_dataset.GameCartridgeDiscriminatorDataset(
                list(zip(data_processor.x_train, data_processor.y_train))
            )
            self.valid_dataset = gcd_dataset.GameCartridgeDiscriminatorDataset(
                list(zip(data_processor.x_eval, data_processor.y_eval))
            )

        if stage == "validate":
            self.valid_dataset = gcd_dataset.GameCartridgeDiscriminatorDataset(
                list(zip(data_processor.x_eval, data_processor.y_eval))
            )

        if stage == "test":
            self.test_dataset = gcd_dataset.GameCartridgeDiscriminatorDataset(
                list(zip(data_processor.x_test, data_processor.y_test))
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=utils.BATCH_SIZE,
            num_workers=utils.NUM_WORKERS,
            shuffle=True,
            generator=torch.Generator().manual_seed(utils.RANDOM_SEED),
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=utils.BATCH_SIZE,
            num_workers=utils.NUM_WORKERS,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=utils.BATCH_SIZE,
            num_workers=utils.NUM_WORKERS,
            shuffle=False,
        )

    def teardown(self, stage: str) -> None:
        if stage == "fit":
            del self.train_dataset
            del self.valid_dataset
        elif stage == "validate":
            del self.valid_dataset
        else:
            del self.test_dataset
