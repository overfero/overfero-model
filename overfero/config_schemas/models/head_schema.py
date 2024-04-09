from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class HeadConfig:
    _target_: str = MISSING


@dataclass
class SigmoidHeadConfig(HeadConfig):
    _target_: str = "overfero.models.heads.SigmoidHead"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="sigmoid_head_schema",
        group="tasks/models/head",
        node=SigmoidHeadConfig,
    )
