from typing import Optional
from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from overfero.config_schemas.models import backbone_schema, adapter_schema, head_schema


@dataclass
class BaseModelConfig:
    _target_: str = MISSING


@dataclass
class BinaryClassificationModelConfig(BaseModelConfig):
    _target_: str = MISSING
    backbone: backbone_schema.BackboneConfig = MISSING
    adapter: Optional[adapter_schema.AdapterConfig] = None
    head: head_schema.HeadConfig = MISSING


def setup_config() -> None:
    backbone_schema.setup_config()
    adapter_schema.setup_config()
    head_schema.setup_config()
    cs = ConfigStore.instance()
    cs.store(
        name="binary_classification_model_schema",
        group="tasks/models",
        node=BinaryClassificationModelConfig,
    )
