import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class Head(Layer):
    pass


class SigmoidHead(Head):
    def __init__(self) -> None:
        super().__init__()
        self.head = Dense(1, activation="sigmoid")

    def call(self, x):
        return self.head(x)
