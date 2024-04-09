from typing import Any, Callable, Optional, Protocol

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from overfero.models.transformations import HuggingFaceTokenizationTransformation, Transformation
import pandas as pd


class DataModule:
    def __init__(
        self,
        batch_size: int,
        text_column_name: str,
        label_column_name: str,
        shuffle: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def initialize_dataloader(self, dataset) -> tf.data.Dataset:
        data_encodings = self.collate_fn(dataset[self.text_column_name])
        data_encodings = tf.data.Dataset.from_tensor_slices(dict(data_encodings))
        data_labels = tf.data.Dataset.from_tensor_slices(dataset[self.label_column_name])
        dataset = tf.data.Dataset.zip(data_encodings, data_labels)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=2000)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset


# class PartialDataModuleType(Protocol):
#     def __call__(self, transformation: Transformation) -> DataModule:
#         pass


class TextClassificationDataModule(DataModule):
    def __init__(
        self,
        train_df_path: str,
        dev_df_path: str,
        test_df_path: str,
        transformation: HuggingFaceTokenizationTransformation,
        text_column_name: str,
        label_column_name: str,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        def tokenization_collate_fn(texts):
            encodings = transformation(texts.to_list())
            return encodings

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            text_column_name=text_column_name,
            label_column_name=label_column_name,
        )

        self.collate_fn = tokenization_collate_fn
        self.train_df_path = train_df_path
        self.dev_df_path = dev_df_path
        self.test_df_path = test_df_path

        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_dataset = pd.read_parquet(self.train_df_path)
            train_dataset = train_dataset[[self.text_column_name, self.label_column_name]]
            self.train_dataset = train_dataset
            dev_dataset = pd.read_parquet(self.dev_df_path)
            dev_dataset = dev_dataset[[self.text_column_name, self.label_column_name]]
            self.dev_dataset = dev_dataset
        elif stage == "test":
            test_dataset = pd.read_parquet(self.test_df_path)
            test_dataset = test_dataset[[self.text_column_name, self.label_column_name]]
            self.test_dataset = test_dataset
