from typing import TYPE_CHECKING
from overfero.trainer.trainers import Trainer
from overfero.training.tasks.bases import TrainingTask
from overfero.utils.io_utils import is_file
from overfero.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility
import tensorflow as tf

if TYPE_CHECKING:
    from overfero.config_schemas.config_schema import Config


class CommonTrainingTask(TrainingTask):
    def __init__(
        self, name: str, trainer: Trainer, best_training_checkpoint: str, last_training_checkpoint: str
    ) -> None:
        super().__init__(name, trainer, best_training_checkpoint, last_training_checkpoint)

    def run(self, config: "Config", task_config) -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        with activate_mlflow(experiment_name, run_id, run_name) as _:
            if "0" in tf.test.gpu_device_name():
                log_artifacts_for_reproducibility()
                # log_training_hparams()
            if is_file(self.last_training_checkpoint):
                self.logger.info(f"Found checkpoint here: {self.last_training_checkpoint} Resuming training")
                self.trainer.model.load_checkpoint(self.last_training_checkpoint)
