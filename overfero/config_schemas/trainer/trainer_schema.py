from dataclasses import field
from omegaconf import MISSING
from pydantic.dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from overfero.config_schemas.trainer import callback_schema
from overfero.config_schemas.training import optimizer_schema, loss_function_schema
from overfero.config_schemas.models import model_schema
from overfero.config_schemas import data_module_schema
from overfero.config_schemas.trainer import logger_schema


@dataclass
class TrainerConfig:
    _target_: str = "overfero.trainer.trainers.Trainer"
    data_modules: data_module_schema.DataModuleConfig = MISSING
    model: model_schema.BaseModelConfig = MISSING
    loss: loss_function_schema.LossFunctionConfig = MISSING
    optimizer: optimizer_schema.BaseOptimizerConfig = MISSING
    callbacks: list[callback_schema.CallbackConfig] = field(default_factory=lambda: [])
    metrics: list[str] = field(default_factory=lambda: ["accuracy"])


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="trainer_schema",
        group="tasks/trainer",
        node=TrainerConfig,
    )
