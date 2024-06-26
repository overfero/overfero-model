from abc import abstractmethod
from typing import Optional
from overfero.models.adapters import Adapter
from overfero.models.backbones import BackBone
from overfero.models.heads import Head
from overfero.models.transformations import Transformation
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras import Model


class BaseModel(Model):
    @abstractmethod
    def get_transformation(self) -> Transformation:
        pass


class BinaryTextClassificationModel(BaseModel):
    def __init__(
        self,
        backbone: BackBone,
        adapter: Optional[Adapter],
        head: Head,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def call(self, x):
        x = self.backbone(x)[1]
        x = self.adapter(x)
        x = self.head(x)
        return x

    def get_transformation(self) -> Transformation:
        return self.backbone.get_transformation()
