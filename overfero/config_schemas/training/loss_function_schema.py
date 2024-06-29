from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class LossFunctionConfig:
    _target_: str = MISSING


@dataclass
class BinaryCrossEntropyConfig(LossFunctionConfig):
    _target_: str = "overfero.training.loss_functions.BinaryCrossEntropyLoss"
    from_logits: bool = False
    label_smoothing: float = 0


@dataclass
class CategoricalCrossEntropyConfig(LossFunctionConfig):
    _target_: str = "overfero.training.loss_functions.CategoricalCrossEntropyLoss"
    from_logits: bool = False
    label_smoothing: float = 0


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_cross_entropy_schema",
        group="tasks/training/loss",
        node=BinaryCrossEntropyConfig,
    )
    cs.store(name="categorical_cross_entropy_schema", group="tasks/training/loss", node=CategoricalCrossEntropyConfig)
