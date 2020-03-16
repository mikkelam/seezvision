import pickle
from pathlib import Path

import pandas as pd
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator
AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_image_paths():
    cache = Path(__file__).parent / 'image_paths.pickle'
    if cache.exists():
        return pickle.load(cache.open('rb'))
    data_dir = Path(__file__).parent / 'data'
    image_paths = list(data_dir.glob('*/*.jpg'))
    ad_ids = [str(i.parent.name) for i in image_paths]
    df = pd.DataFrame({'path': map(str, image_paths), 'ad_id': ad_ids})
    pickle.dump(df, cache.open('wb'))

    return df


def make_labelset():
    df = pd.read_csv('cars.csv', usecols=[0, 1, 2, 3],
                     names=['ad_id', 'make', 'model', 'submodel'])
    submodel_cars = {'Mercedes-Benz', 'Lexus', 'BMW'}
    df = df[~pd.isnull(df.model)]
    df = df[((df.make.isin(submodel_cars)) &
             (~pd.isnull(df.submodel)) |
             ~df.make.isin(submodel_cars)
             )
    ]

    def labellize(row):
        return ' '.join(row)

    df.loc[df['make'].isin(submodel_cars), 'label'] = df.loc[df['make'].isin(submodel_cars)][
        ['make', 'submodel']].apply(labellize, axis=1)
    df.loc[~df.make.isin(submodel_cars), 'label'] = df.loc[~df.make.isin(submodel_cars)][
        ['make', 'model']].apply(labellize, axis=1)
    df.drop(['make', 'model', 'submodel'], axis=1, inplace=True)
    return df


def dataset():
    image_df = get_image_paths()
    labelset = make_labelset()

    df = image_df.merge(labelset, on='ad_id')

    datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.2, samplewise_center=True,
                                 samplewise_std_normalization=True)
    # datagen.mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3)) # ordering: [R, G, B]
    # datagen.std = 64.
    params = {'dataframe': df, 'x_col': 'path', 'y_col': 'label', 'target_size': (256, 256),
              'class_mode': 'categorical', 'batch_size': 64, 'shuffle': True, 'seed': 42}

    train_gen = datagen.flow_from_dataframe(**params, subset='training')
    val_gen = datagen.flow_from_dataframe(**params, subset='validation')

    # Grab our output shapes and dtype
    # images, labels = next(train_gen)
    # print(images.dtype, images.shape) # float32 (64, 256, 256, 3)
    # print(labels.dtype, labels.shape) # float32 (64, 927)

    ds_train = tf.data.Dataset.from_generator(
        lambda: train_gen,  # stupid tf wants a callable, here u go idiots
        output_types=(tf.float32, tf.float32),
        output_shapes=([64, 256, 256, 3], [64, 927])
    )
    ds_val = tf.data.Dataset.from_generator(
        lambda: val_gen,  # stupid tf wants a callable, here u go idiots
        output_types=(tf.float32, tf.float32),
        output_shapes=([64, 256, 256, 3], [64, 927])
    )
    return ds_train, ds_val
    # val_gen = datagen.flow_from_dataframe(**params, subset='validation')
    # datagen.fit(ds)
    # ...


if __name__ == '__main__':
    dataset()
