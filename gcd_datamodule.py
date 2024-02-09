from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import gcd_dataset
import utils
from data_processor import DataProcessor


class GameCartridgeDiscriminatorDatamodule(LightningDataModule):
    """Datamodule for game cartridge discriminator dataset."""

    def __init__(self, samples_path):
        """Init function for game cartridge discriminator datamodule

        Args:
            training_data (List[tuple]): List of tuple (image,label)
            valid_data (List[tuple]): List of tuple (image,label)
            test_data (List[tuple]): List of tuple (image,label)
        """
        super().__init__()

        self.samples_path = samples_path

    def setup(self, stage: str):
        data_processor = DataProcessor(self.samples_path, 0.3, 0)
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
            # collate_fn=utils.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=utils.BATCH_SIZE,
            num_workers=utils.NUM_WORKERS,
            shuffle=False,
            # collate_fn=utils.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=utils.BATCH_SIZE,
            num_workers=utils.NUM_WORKERS,
            shuffle=False,
            # collate_fn=utils.collate_fn
        )

    def teardown(self, stage: str) -> None:
        if stage == "fit":
            del self.train_dataset
            del self.valid_dataset
        elif stage == "validate":
            del self.valid_dataset
        else:
            del self.test_dataset
