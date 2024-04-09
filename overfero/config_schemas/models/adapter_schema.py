from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class AdapterConfig:
    _target_: str = MISSING


@dataclass
class DenseAdapterConfig(AdapterConfig):
    _target_: str = "overfero.models.adapters.DenseAdapter"
    output_dim: int = MISSING
    activation_fn: str = "relu"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="dense_adapter_schema",
        group="tasks/models/adapter",
        node=DenseAdapterConfig,
    )
