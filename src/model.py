
import tensorflow as tf
import efficientnet.tfkeras as efn

def create_model(input_shape, output_shape) -> tf.keras.Model:
    model = efn.EfficientNetB2(input_shape=input_shape, weights='noisy-student', include_top=False)
    model = tf.keras.Sequential([
        model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(output_shape)
    ])
    return model
