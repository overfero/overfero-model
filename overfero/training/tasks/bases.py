from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


from overfero.trainer.trainers import Trainer
from overfero.utils.utils import get_logger

if TYPE_CHECKING:
    from overfero.config_schemas.config_schema import Config


class TrainingTask(ABC):
    def __init__(
        self, name: str, trainer: Trainer, best_training_checkpoint: str, last_training_checkpoint: str
    ) -> None:
        super().__init__()
        self.name = name
        self.trainer = trainer
        self.best_training_checkpoint = best_training_checkpoint
        self.last_training_checkpoint = last_training_checkpoint
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def run(self, config: "Config", task_config) -> None:
        pass
