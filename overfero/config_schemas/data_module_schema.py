from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from overfero.config_schemas.models import transformation_schema


@dataclass
class DataModuleConfig:
    _target_: str = MISSING
    batch_size: int = MISSING
    text_column_name: str = "cleaned_text"
    label_column_name: str = "label"
    shuffle: bool = False


class TextClassificationDataModuleConfig(DataModuleConfig):
    _target_: str = "overfero.data_modules.data_modules.TextClassificationDataModule"
    train_df_path: str = MISSING
    dev_df_path: str = MISSING
    test_df_path: str = MISSING
    transformation: transformation_schema.TransformationConfig = MISSING


def setup_config() -> None:
    transformation_schema.setup_config()
    cs = ConfigStore.instance()
    cs.store(
        name="text_classification_data_module_schema",
        group="tasks/data_module",
        node=TextClassificationDataModuleConfig,
    )
