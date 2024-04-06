import pandas as pd
import tensorflow as tf


class TextClassificationDataset(tf.data.Dataset):
    def __init__(self, df_path: str, text_column_name: str, label_column_name: str) -> None:
        super(TextClassificationDataset, self).__init__(
            variant_tensor=None
        )  # Call the __init__() method of tf.data.Dataset with the required variant_tensor argument.
        self.df = pd.read_parquet(df_path)
        self.text_column_name = text_column_name
        self.label_column_name = label_column_name

    def _inputs(self) -> list:
        return []

    def element_spec(self) -> tuple:
        return (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.int32))

    def __getitem__(self, idx: int) -> tuple[str, tf.Tensor]:
        row = self.df.iloc[idx]

        text = row[self.text_column_name]
        label = row[self.label_column_name]

        return text, tf.constant([label])

    def __len__(self) -> int:
        return len(self.df)

    def _flat_tensor_specs(self):
        return self.element_spec()
