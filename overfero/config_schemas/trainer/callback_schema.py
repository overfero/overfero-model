from omegaconf import MISSING, SI
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
class ModelCheckpointConfig(CallbackConfig):
    _target_: str = "tensorflow.keras.callbacks.ModelCheckpoint"
    filepath: str = "./data/tensorflow/checkpoints/"
    monitor: str = "val_accuracy"
    verbose: int = 0
    save_best_only: bool = True
    save_weights_only: bool = False
    mode: str = "max"
    save_freq: str = "epoch"


@dataclass
class BestModelCheckpointConfig(ModelCheckpointConfig):
    filepath: str = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/")


@dataclass
class LastModelCheckpointConfig(ModelCheckpointConfig):
    filepath: str = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/")
    save_best_only: bool = False
    save_weights_only: bool = True


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
        name="model_checkpoint_schema",
        group="tasks/trainer/callback",
        node=ModelCheckpointConfig,
    )
    cs.store(
        name="best_model_checkpoint_schema",
        group="tasks/trainer/callback",
        node=BestModelCheckpointConfig,
    )
    cs.store(
        name="last_model_checkpoint_schema",
        group="tasks/trainer/callback",
        node=LastModelCheckpointConfig,
    )
    cs.store(
        name="early_stopping_schema",
        group="tasks/trainer/callback",
        node=EarlyStoppingConfig,
    )
    cs.store(
        name="tensorboard_schema",
        group="tasks/trainer/callback",
        node=TensorBoardConfig,
    )
