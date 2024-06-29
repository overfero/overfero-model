from os import name
from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from overfero.config_schemas.trainer import trainer_schema


@dataclass
class TaskConfig:
    _target_: str = MISSING
    name: str = MISSING
    trainer: trainer_schema.TrainerConfig = MISSING
