from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class MLFlowLoggerConfig(LoggerConfig):
    _target_: str = "mlflow.tensorflow.MLflowCallback"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="mlflow_logger_schema",
        group="tasks/trainer/logger",
        node=MLFlowLoggerConfig,
    )
