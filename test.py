import pandas as pd
from overfero.data_modules.data_modules import TextClassificationDataModule
from overfero.data_modules.datasets import TextClassificationDataset
from overfero.models.transformations import HuggingFaceTokenizationTransformation


transformation = HuggingFaceTokenizationTransformation(
    pretrained_tokenizer_name_or_path="trained_tokenizer",
    max_sequence_length=128,
)

dataset = TextClassificationDataset(
    df_path="data/processed/test.parquet",
    text_column_name="cleaned_text",
    label_column_name="label",
)
print("_" * 100)
print("Tokenizer = ", transformation)
print("_" * 100)
print("Dataset = ", dataset[1])
print("_" * 100)
print(transformation([dataset[1][0]]))

data_module = TextClassificationDataModule(
    train_df_path="data/processed/train.parquet",
    dev_df_path="data/processed/dev.parquet",
    test_df_path="data/processed/test.parquet",
    transformation=transformation,
    text_column_name="cleaned_text",
    label_column_name="label",
    batch_size=32,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)

data_module.setup()
print("_" * 100)
print("DataModule = ", data_module)
print("_" * 100)
