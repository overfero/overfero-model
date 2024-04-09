from tensorflow.keras.layers import Layer
from transformers import AutoConfig, TFAutoModel, BatchEncoding
import tensorflow as tf
from overfero.models.transformations import Transformation
from overfero.utils.io_utils import translate_gcs_dir_to_local


class BackBone(Layer):
    pass


class HuggingFaceBackbone(BackBone):
    def __init__(self, pretrained_model_name_or_path: str, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = self.get_backbone(pretrained_model_name_or_path, pretrained)

    def call(self, x):
        output = self.backbone(x)
        return output

    def get_backbone(self, pretrained_model_name_or_path: str, pretrained: bool):
        path = translate_gcs_dir_to_local(pretrained_model_name_or_path)
        config = AutoConfig.from_pretrained(path)
        if pretrained:
            backbone_from_pretrained = TFAutoModel.from_pretrained(path, config=config)
            return backbone_from_pretrained

        backbone_from_config = TFAutoModel.from_config(config)
        return backbone_from_config
