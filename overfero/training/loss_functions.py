import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Loss


class BaseLoss(Loss):
    pass


class BinaryCrossEntropyLoss(BaseLoss):
    def __init__(self, from_logits: bool = False, label_smoothing: float = 0):
        super().__init__()
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.loss = BinaryCrossentropy(from_logits=from_logits, label_smoothing=label_smoothing)

    def call(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
