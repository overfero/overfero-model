from typing import Any, Callable, Optional, Protocol

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from overfero.models.transformations import HuggingFaceTokenizationTransformation, Transformation
import pandas as pd


class DataModule:
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

    def initialize_dataloader(self, dataset: Sequence) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_generator(
            generator=dataset,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.string),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )
        dataset = dataset.batch(
            batch_size=self.batch_size,
            drop_remainder=self.drop_last,
        )
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(dataset))
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


class PartialDataModuleType(Protocol):
    def __call__(self, transformation: Transformation) -> DataModule:
        pass


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
        num_workers: int = 0,
        drop_last: bool = False,
    ) -> None:
        @tf.function
        def tokenization_collate_fn(texts, labels) -> tuple[tf.Tensor, tf.Tensor]:
            print("check5")
            encodings = transformation(list(texts))
            print("check6")
            return encodings, labels

        super().__init__(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
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
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (train_dataset[self.text_column_name].values, train_dataset[self.label_column_name].values)
            )

            self.train_dataset = train_dataset.map(self.collate_fn)


# from torch import Tensor
# from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, default_collate
# from transformers import BatchEncoding

# from cybulde.data_modules.datasets import TextClassificationDataset
# from cybulde.models.transformations import HuggingFaceTokenizationTransformation, Transformation


# class DataModule(LightningDataModule):
#     def __init__(
#         self,
#         batch_size: int,
#         shuffle: bool = False,
#         sampler: Optional[Sampler] = None,
#         batch_sampler: Optional[BatchSampler] = None,
#         num_workers: int = 0,
#         collate_fn: Optional[Callable[[Any], Any]] = None,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         persistent_workers: bool = False,
#     ) -> None:
#         super().__init__()

#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.sampler = sampler
#         self.batch_sampler = batch_sampler
#         self.num_workers = num_workers
#         self.collate_fn = collate_fn
#         self.pin_memory = pin_memory
#         self.drop_last = drop_last
#         self.persistent_workers = persistent_workers

#     def initialize_dataloader(self, dataset: Dataset, is_test: bool) -> DataLoader:
#         return DataLoader(
#             dataset,
#             batch_size=self.batch_size,
#             shuffle=self.shuffle and not is_test,
#             sampler=self.sampler,
#             batch_sampler=self.batch_sampler,
#             num_workers=self.num_workers,
#             collate_fn=self.collate_fn,
#             pin_memory=self.pin_memory,
#             drop_last=self.drop_last,
#             persistent_workers=self.persistent_workers,
#         )


# class PartialDataModuleType(Protocol):
#     def __call__(self, transformation: Transformation) -> DataModule: ...


# class TextClassificationDataModule(DataModule):
#     def __init__(
#         self,
#         train_df_path: str,
#         dev_df_path: str,
#         test_df_path: str,
#         transformation: HuggingFaceTokenizationTransformation,
#         text_column_name: str,
#         label_column_name: str,
#         batch_size: int,
#         shuffle: bool = False,
#         sampler: Optional[Sampler] = None,
#         batch_sampler: Optional[BatchSampler] = None,
#         num_workers: int = 0,
#         pin_memory: bool = False,
#         drop_last: bool = False,
#         persistent_workers: bool = False,
#     ) -> None:
#         def tokenization_collate_fn(batch: list[tuple[str, int]]) -> tuple[BatchEncoding, Tensor]:
#             texts, labels = default_collate(batch)
#             encodings = transformation(texts)
#             return encodings, labels

#         super().__init__(
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             batch_sampler=batch_sampler,
#             num_workers=num_workers,
#             collate_fn=tokenization_collate_fn,
#             pin_memory=pin_memory,
#             drop_last=drop_last,
#             persistent_workers=persistent_workers,
#         )

#         self.train_df_path = train_df_path
#         self.dev_df_path = dev_df_path
#         self.test_df_path = test_df_path

#         self.text_column_name = text_column_name
#         self.label_column_name = label_column_name

#     def setup(self, stage: Optional[str] = None) -> None:
#         if stage == "fit" or stage is None:
#             self.train_dataset = TextClassificationDataset(
#                 self.train_df_path, self.text_column_name, self.label_column_name
#             )
#             self.dev_dataset = TextClassificationDataset(
#                 self.dev_df_path, self.text_column_name, self.label_column_name
#             )

#         if stage == "test":
#             self.test_dataset = TextClassificationDataset(
#                 self.test_df_path, self.text_column_name, self.label_column_name
#             )

#     def train_dataloader(self) -> DataLoader:
#         return self.initialize_dataloader(self.train_dataset, is_test=False)

#     def val_dataloader(self) -> DataLoader:
#         return self.initialize_dataloader(self.dev_dataset, is_test=True)

#     def test_dataloader(self) -> DataLoader:
#         return self.initialize_dataloader(self.test_dataset, is_test=True)
