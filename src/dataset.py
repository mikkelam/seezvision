import math
import pickle
from pathlib import Path

from joblib import Parallel, delayed
import pandas as pd
import numpy  as np
import tensorflow as tf
import tensorflow_addons as tfa
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.utils import compute_class_weight, shuffle
from tensorflow import one_hot
from tensorflow.python.data.experimental import cardinality
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator
from tqdm import tqdm

AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed = 31


class Dataset:
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data'
    record_dir = project_dir / 'records'
    path_cache = project_dir / 'image_paths.pickle'
    labels = project_dir / 'labels.txt'
    class_weight = project_dir / 'class_weights.pickle'
    norm_weights = [np.array([0.4797589, 0.47081122, 0.45555392]), np.array([0.08123032, 0.07896693, 0.07895421])]
    IMAGE_SIZE = 224

    @classmethod
    def get_image_paths(cls):
        if cls.path_cache.exists():
            return pickle.load(cls.path_cache.open('rb'))
        print('Finding all image paths')
        image_paths = list(cls.data_dir.glob('*/*.jpg'))
        ad_ids = [str(i.parent.name) for i in image_paths]
        df = pd.DataFrame({'path': map(str, image_paths), 'ad_id': ad_ids})
        pickle.dump(df, cls.path_cache.open('wb'))

        return df

    @classmethod
    def find_corrupt_images(cls, delete=False):
        from PIL import Image
        from tqdm import tqdm
        image_df = cls.get_image_paths()

        def check_image(image_path):
            try:
                im = Image.open(image_path)
                im.load()
            except:
                s = f'{"Deleting" if delete else "Found"} corrupt image: {image_path}'
                if delete:
                    Path(image_path).unlink()
                print(s)

        tasks = tqdm(image_df.path.tolist())
        results = Parallel(n_jobs=7)(delayed(check_image)(img) for img in tasks)

    @classmethod
    def write(cls, samples_per_class=3500, sample_class_cutoff=50):
        def write_shard(df: pd.DataFrame, shard_path: Path):
            with tf.io.TFRecordWriter(str(shard_path)) as writer:
                for row in df.itertuples():
                    image_path, label = row.path, row.label
                    with open(image_path, 'rb') as img:
                        feature = {
                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.read()]))
                        }
                        example = tf.train.Example(features=tf.train.Features(feature=feature))
                        writer.write(example.SerializeToString())

        image_df = cls.get_image_paths()
        labelset = cls.make_labelset()

        df = image_df.merge(labelset, on='ad_id')

        df = df.groupby('label').head(samples_per_class)
        df = df[df.groupby('label')['label'].transform('size') > sample_class_cutoff]
        df = shuffle(df)

        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df.label.values)
        cls.labels.write_text("\n".join(label_encoder.classes_))
        class_weights = dict(enumerate(compute_class_weight('balanced',
                                                            np.unique(df.label),
                                                            df.label)))
        pickle.dump(cls.class_weight.open('rb'))
        train, val = train_test_split(df, test_size=0.15)
        tasks = []
        for key, df in {'train': train, "val": val}.items():
            image_count = df.shape[0]
            images_per_shard = 20000.0
            shards = int(math.ceil(image_count / images_per_shard))
            df_chunks = np.array_split(df, shards)
            tasks += [(df_chunks[shard], cls.record_dir / f'{key}-images{str(shard)}.tfrecord')
                      for shard in range(shards)]

        results = Parallel(n_jobs=20)(delayed(write_shard)(*task) for task in tqdm(tasks))

    @staticmethod
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

    @classmethod
    def read(cls, batch_size):
        labels = cls.labels.read_text()
        class_weight = pickle.load(cls.class_weight.open('rb'))
        N_CLASSES = len(class_weight.keys())

        train = list(map(str, cls.record_dir.glob('train*.tfrecord')))
        val = list(map(str, cls.record_dir.glob('val*.tfrecord')))
        ds_val = tf.data.TFRecordDataset(
            val, num_parallel_reads=AUTOTUNE
        )
        ds_train = tf.data.TFRecordDataset(
            train, num_parallel_reads=AUTOTUNE
        )

        image_feature_description = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
        }

        def parse(example, train=True):
            example = tf.io.parse_single_example(example, image_feature_description)
            label = example['label']
            img = example['image']
            img = tf.io.decode_jpeg(img)
            img = (tf.cast(img, tf.float32) / 255)
            # img = tf.image.resize_with_crop_or_pad(img, cls.IMAGE_SIZE, cls.IMAGE_SIZE)
            img = tf.image.resize_with_pad(img, cls.IMAGE_SIZE, cls.IMAGE_SIZE, antialias=False)
            # img = tf.image.resize(img, (cls.IMAGE_SIZE, cls.IMAGE_SIZE))
            # if train:
            #     img = tf.image.random_brightness(img, 0.2)
            #     img = tf.image.random_flip_left_right(img)
            #     img = tf.image.random_flip_up_down(img)

            label = one_hot(label, N_CLASSES)
            return img, label

        ds_val = ds_val.map(lambda x: parse(x, train=False), num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.map(parse, num_parallel_calls=AUTOTUNE)

        ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
        ds_val = ds_val.batch(batch_size).prefetch(AUTOTUNE)

        return ds_train, ds_val,  class_weight, labels, cls.IMAGE_SIZE, N_CLASSES

    @classmethod
    def normalize_layer(cls, input_shape) -> tf.keras.layers.experimental.preprocessing.Normalization:
        n = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1, input_shape=input_shape)
        if cls.norm_weights:
            n.build(input_shape)
            n.set_weights(cls.norm_weights)
            return n
        ds_train, _, _, _, _, _ = cls.read(batch_size=22)
        images = ds_train.map(lambda img, label: img)
        n = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        n.adapt(images)
        print(n.get_weights())
        return n


if __name__ == '__main__':
    # Dataset.write(samples_per_class=1500)
    Dataset.normalize_layer()
    # print(next(Dataset.read(24)[0].take(1).as_numpy_iterator()))
    # dataset2()
    # find_corrupt_images()
