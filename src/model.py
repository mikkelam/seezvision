
import tensorflow as tf
import efficientnet.tfkeras as efn

# from dataset import Dataset
from src.dataset import Dataset


def create_model(input_shape, output_shape) -> tf.keras.Model:
    model = efn.EfficientNetB2(input_shape=input_shape, weights='noisy-student', include_top=False)
    model = tf.keras.Sequential([
        Dataset.normalize_layer(input_shape),
        tf.keras.layers.experimental.preprocessing.RandomFlip(),
        tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.3),
        tf.keras.layers.experimental.preprocessing.RandomContrast(0.35),
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(output_shape)
    ])
    return model
