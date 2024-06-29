from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class BaseOptimizerConfig:
    _target_: str = MISSING


@dataclass
class AdamOptimizerConfig(BaseOptimizerConfig):
    _target_: str = "tensorflow.keras.optimizers.Adam"
    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="adam_optimizer_schema",
        group="task/training/optimizer",
        node=AdamOptimizerConfig,
    )
