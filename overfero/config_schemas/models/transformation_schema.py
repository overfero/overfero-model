from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class TransformationConfig:
    _target_: str = MISSING


@dataclass
class HuggingFaceTokenizationTransformationConfig(TransformationConfig):
    _target_: str = "overfero.models.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length: int = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_tokenization_transformation_schema",
        group="tasks/models/transformation",
        node=HuggingFaceTokenizationTransformationConfig,
    )
