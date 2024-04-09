import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class Adapter(Layer):
    pass


class DenseAdapter(Adapter):
    def __init__(self, output_dim: int, activation_fn: str) -> None:
        super().__init__()
        self.dense_1 = Dense(output_dim, activation=activation_fn)
        self.dense_2 = Dense(output_dim // 4, activation=activation_fn)
        self.dense_3 = Dense(output_dim // 16, activation=activation_fn)

    def call(self, x):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x
