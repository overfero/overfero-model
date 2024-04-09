from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class CallbackConfig:
    _target_: str = MISSING


@dataclass
class EarlyStoppingConfig(CallbackConfig):
    _target_: str = "tensorflow.keras.callbacks.EarlyStopping"
    monitor: str = "val_accuracy"
    min_delta: float = 0.0
    patience: int = 0
    verbose: int = 0
    mode: str = "max"
    baseline: float = None
    restore_best_weights: bool = False


@dataclass
class TensorBoardConfig(CallbackConfig):
    _target_: str = "tensorflow.keras.callbacks.TensorBoard"
    log_dir: str = "tensorboard_logs"
    histogram_freq: int = 0
    write_graph: bool = True
    write_images: bool = False
    update_freq: str = "epoch"
    profile_batch: int = 2
    embeddings_freq: int = 0
    embeddings_metadata: str = None
    embeddings_data: str = None


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="early_stopping_schema",
        node=EarlyStoppingConfig,
    )
    cs.store(
        name="tensorboard_schema",
        node=TensorBoardConfig,
    )
