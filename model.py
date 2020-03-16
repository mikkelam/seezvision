import tensorflow as tf

import efficientnet.tfkeras as efn

from seez_image_gen import dataset

IMG_SHAPE = (256, 256, 3)
OUTPUT_SHAPE = 927

ds_train, ds_val = dataset()
model = efn.EfficientNetB2(input_shape=IMG_SHAPE, weights='noisy-student', include_top=False)

model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(OUTPUT_SHAPE)
])

base_learning_rate = 0.001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5)])
model.summary()
imgs, labels = next(ds_train.as_numpy_iterator())
...
history = model.fit(ds_train,
                         epochs=10,
                         # initial_epoch =  0,
                         validation_data=ds_val)


# def add_fc_layer():
#     global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#     feature_batch_average = global_average_layer(feature_batch)
#     print(feature_batch_average.shape)
#
#

# feature_batch = model(imgs)


# print(feature_batch.shape)

...
