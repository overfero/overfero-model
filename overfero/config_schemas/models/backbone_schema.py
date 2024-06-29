from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from overfero.config_schemas.models.transformation_schema import TransformationConfig


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    transformation: TransformationConfig = MISSING


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "overfero.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/models/backbone",
        node=HuggingFaceBackboneConfig,
    )
