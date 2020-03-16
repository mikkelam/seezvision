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
    data_dir = Path(__file__).parent / 'data'
    record_dir = Path(__file__).parent / 'records'
    IMAGE_SIZE = 223

    @classmethod
    def get_image_paths(cls):
       cache = Path(__file__).parent / 'image_paths.pickle'
       if cache.exists():
           return pickle.load(cache.open('rb'))
       print('Finding all image paths')
       image_paths = list(cls.data_dir.glob('*/*.jpg'))
       ad_ids = [str(i.parent.name) for i in image_paths]
       df = pd.DataFrame({'path': map(str, image_paths), 'ad_id': ad_ids})
       pickle.dump(df, cache.open('wb'))

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
        pickle.dump(label_encoder, open('label_encoder.pickle', 'wb'))
        Path('labels.txt').write_text("\n".join(label_encoder.classes_))
        class_weights = dict(enumerate(compute_class_weight('balanced',
                                             np.unique(df.label),
                                             df.label)))
        pickle.dump(class_weights,open('class_weights.pickle', 'wb'))
        exit()
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
        le = pickle.load(open('label_encoder.pickle', 'rb'))
        class_weight = pickle.load(open('class_weights.pickle','rb'))
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
            img = tf.image.decode_jpeg(img)
            img = (tf.cast(img, tf.float32) / 255)
            # img = tf.image.random_crop(img, size=(cls.IMAGE_SIZE, cls.IMAGE_SIZE, 3))
            img = tf.image.resize(img, (cls.IMAGE_SIZE, cls.IMAGE_SIZE))
            img = tf.image.per_image_standardization(img)
            if train:
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)

            label = one_hot(label, N_CLASSES)
            return img, label


        ds_val = ds_val.map(lambda x: parse(x, train=False), num_parallel_calls=AUTOTUNE)
        ds_train = ds_train.map(parse,  num_parallel_calls=AUTOTUNE)

        ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)
        ds_val = ds_val.batch(batch_size).prefetch(AUTOTUNE)



        return ds_train, ds_val, class_weight, le, cls.IMAGE_SIZE, N_CLASSES




def dataset():
    IMAGE_SIZE = 224

    datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.15, samplewise_center=True,
                                 samplewise_std_normalization=True, horizontal_flip=True, vertical_flip=True,
                                 rotation_range=45)
    params = {'dataframe': df, 'x_col': 'path', 'y_col': 'label', 'target_size': (IMAGE_SIZE, IMAGE_SIZE),
              'class_mode': 'categorical', 'batch_size': BATCH_SIZE, 'shuffle': True, 'seed': 42}

    train_gen = datagen.flow_from_dataframe(**params, subset='training')
    val_gen = datagen.flow_from_dataframe(**params, subset='validation')

    # Grab our output shapes and dtype
    # images, labels = next(train_gen)
    # print(images.dtype, images.shape) # float32 (64, 256, 256, 3)
    # print(labels.dtype, labels.shape) # float32 (64, 927)
    output_types = (tf.float32, tf.float32)
    output_shapes = ([None, IMAGE_SIZE, IMAGE_SIZE, 3], [None, N_CLASSES])

    ds_train = tf.data.Dataset.from_generator(
        lambda: train_gen,  # stupid tf wants a callable, here u go idiots
        output_types=output_types,
        output_shapes=output_shapes,
    )

    ds_val = tf.data.Dataset.from_generator(
        lambda: val_gen,  # stupid tf wants a callable, here u go idiots
        output_types=output_types,
        output_shapes=output_shapes,
    )

    train_steps = len(train_gen) // BATCH_SIZE
    val_steps = len(val_gen) // BATCH_SIZE

    class_weights = compute_class_weight('balanced',
                                         np.unique(train_gen.classes),
                                         train_gen.classes)

    print(
        f'Loaded dataset with {df.shape[0]} samples, {N_CLASSES} classes and train steps:{train_steps}, val steps:{val_steps} in batches of {BATCH_SIZE}  ')
    return ds_train, ds_val, train_steps, val_steps, class_weights, N_CLASSES, BATCH_SIZE, IMAGE_SIZE




if __name__ == '__main__':
    Dataset.write(samples_per_class=1500)
    # print(next(Dataset.read(24)[0].take(1).as_numpy_iterator()))
    # dataset2()
    # find_corrupt_images()
