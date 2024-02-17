from load_data import load_data
from sklearn.model_selection import train_test_split
import time
import random
from pathlib import Path


class DataProcessor:
    """
    Class that builds train, validation and test set.
    """

    def __init__(self, samples_path: Path, kept_size: float, random_state: int) -> None:
        """Data Processor init function.

        Args:
            samples_path (Path): Path to samples csv containing rows with the following structure: sample_path, console {true,false}.
            kept_size (float): Percentage of data to retain for validation/test. The formers will have, each one, half of these data.
            random_state (int): Seed for random split.

        Raises:
            ValueError: If there are no data to load in samples_path
        """

        self.samples = load_data(samples_path)

        if self.samples is None:
            raise ValueError("No samples loaded")

        (
            self.x_train,
            self.y_train,
            self.x_eval,
            self.y_eval,
            self.x_test,
            self.y_test,
        ) = self._data_load_and_process_data(kept_size, random_state)

        self.labels_name = self._extract_labels()

    def _data_load_and_process_data(self, kept_size: float, random_state: int):
        """

        Process training and test data.

        Args:
        kept_size (float): Percentage of the dataset to reatain for validaton and test set.
        random_state (int): Seed for random split.

        Returns:
        tuple: x_train, y_train, x_eval, y_eval, x_test, y_test

        """
        # Build train data and kept set
        x_train, x_kept, y_train, y_kept = train_test_split(
            self.samples[0],
            self.samples[1],
            test_size=kept_size,
            random_state=random_state,
        )
        # Build validation and test set
        x_test, x_eval, y_test, y_eval = train_test_split(
            x_kept,
            y_kept,
            test_size=kept_size / 2,
            random_state=random_state,
        )

        return (
            x_train,
            y_train,
            x_eval,
            y_eval,
            x_test,
            y_test,
        )
