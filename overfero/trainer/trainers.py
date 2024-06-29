from functools import partial

from overfero.data_modules.data_modules import DataModule
from overfero.models.models import BaseModel
from overfero.training.loss_functions import BaseLoss
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.callbacks import Callback


class BaseTrainer:
    pass


class Trainer(BaseTrainer):
    def __init__(
        self,
        strategy: tf.distribute.Strategy,
        data_modules: DataModule,
        model: BaseModel,
        loss: BaseLoss,
        optimizer: Optimizer,
        callbacks: list[Callback],
        metrics: list[str],
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.callbacks = callbacks
        if isinstance(data_modules, partial):
            self.data_modules = data_modules(self.transformations)
        else:
            self.data_modules = data_modules
        self.transformations = model.get_transformation()
        self.train_dataset, self.dev_dataset = self.get_datasets()
        self.model = self.get_model(model, loss, optimizer, metrics)

    def fit(self, epochs: int) -> None:
        self.model.fit(self.train_dataset, validation_data=self.dev_dataset, epochs=epochs, callbacks=self.callbacks)

    def get_model(self, model: BaseModel, loss: BaseLoss, optimizer: Optimizer, metrics: list[str]) -> BaseModel:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def get_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        self.data_modules.setup()
        num_gpus = self.strategy.num_replicas_in_sync
        train_dataset = self.data_modules.initialize_dataloader(
            dataset=self.data_modules.train_dataset, num_gpus=num_gpus
        )
        dev_dataset = self.data_modules.initialize_dataloader(dataset=self.data_modules.dev_dataset, num_gpus=num_gpus)
        train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        dev_dataset = self.strategy.experimental_distribute_dataset(dev_dataset)
        return train_dataset, dev_dataset
