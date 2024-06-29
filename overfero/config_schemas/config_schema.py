from typing import Optional

from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass

from overfero.config_schemas.infrastructure import infrastructure_schema


@dataclass
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = infrastructure_schema.InfrastructureConfig()
    docker_image: Optional[str] = None
    seed: int = 98


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)
