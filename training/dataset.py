# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Multi-resolution input data pipeline."""

import os
import glob
import numpy as np
import functools
import tensorflow as tf
import tflex
import tensorflow_datasets as tfds
import dnnlib
import dnnlib.tflib as tflib

from tensorflow.python.platform import gfile

# ----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.


class TFRecordDataset:
    def __init__(self,
                 tfrecord_dir,               # Directory containing a collection of tfrecords files.
                 split='train',  # Dataset split, 'train' or 'test'
                 from_tfrecords=False,    # Load from tfrecords or from tensorflow datasets
                 tfds_data_dir=None,     # Directory from which tensorflow datasets load
                 resolution=None,     # Dataset resolution, None = autodetect.
                 label_file=None,     # Relative path of the labels file, None = autodetect.
                 max_label_size=0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
                 num_samples=None,     # Maximum number of images to use, None = use all images.
                 num_val_images=None,     # Number of validation images split from the training set, None = use separate validation set.
                 repeat=True,     # Repeat dataset indefinitely?
                 shuffle_mb=4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
                 prefetch_mb=2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
                 buffer_mb=256,      # Read buffer size (megabytes).
                 num_threads=2):       # Number of concurrent threads.

        self.tfrecord_dir = tfrecord_dir
        self.resolution = None
        self.shape = []        # [channels, height, width]
        self.dtype = 'uint8'
        self.dynamic_range = [0, 255]
        self.from_tfrecords = from_tfrecords
        self.label_file = label_file
        self.label_size = None      # components
        self.label_dtype = None
        self.num_samples = num_samples
        self._np_labels = None
        self._tf_minibatch_in = None
        self._tf_labels_var = None
        self._tf_labels_dataset = None
        self._tf_dataset = None
        self._tf_iterator = None
        self._tf_init_op = None
        self._tf_minibatch_np = None
        self._cur_minibatch = -1

        # List tfrecords files and inspect their shapes.
        assert gfile.IsDirectory(self.tfrecord_dir)
        tfr_files = sorted(tf.io.gfile.glob(os.path.join(self.tfrecord_dir, '*.tfrecords')))
        assert len(tfr_files) >= 1

        # To avoid parsing the entire dataset looking for the max
        # resolution, just assume that we can extract the LOD level
        # from the tfrecords file name.
        tfr_shapes = []
        lod = -1
        for tfr_file in tfr_files:
          match = re.match('.*([0-9]+).tfrecords', tfr_file)
          if match:
            level = int(match.group(1))
            res = 2**level
            tfr_shapes.append((3, res, res))

        # Determine shape and resolution.
        max_shape = max(tfr_shapes, key=np.prod)
        self.resolution = resolution if resolution is not None else max_shape[1]
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = [max_shape[0], self.resolution, self.resolution]
        tfr_lods = [self.resolution_log2 - int(np.log2(shape[1])) for shape in tfr_shapes]
        assert all(shape[0] == max_shape[0] for shape in tfr_shapes)
        assert all(shape[1] == shape[2] for shape in tfr_shapes)
        assert all(shape[1] == self.resolution // (2**lod) for shape, lod in zip(tfr_shapes, tfr_lods))
        assert all(lod in tfr_lods for lod in range(self.resolution_log2 - 1))

        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<30, 0], dtype=np.float32)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_images is not None and self._np_labels.shape[0] > max_images:
            self._np_labels = self._np_labels[:max_images]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name

        # Build TF expressions.
        with tf.name_scope('Dataset'), tflex.device('/cpu:0'):
            self._tf_minibatch_in = tf.placeholder(tf.int64, name='minibatch_in', shape=[])
            if num_samples is not None:
                dset = dset.take(self.num_samples)
            if self.from_tfrecords:
                dset = dset.map(functools.partial(self.parse_tfrecord_tf, resize=resize), num_parallel_calls=num_threads)
                self._tf_labels_var = tflib.create_var_with_large_initial_value(self._np_labels, name='labels_var')
                self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)
                dset = tf.data.Dataset.zip((dset, self._tf_labels_dataset))
            else:
                dset = dset.map(functools.partial(self.parse_tfdataset_tf, resize=resize), num_parallel_calls=num_threads)
            bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
            if shuffle_mb > 0:
                dset = dset.shuffle(((shuffle_mb << 20) - 1) // bytes_per_item + 1)
            if repeat:
                dset = dset.repeat()
            if prefetch_mb > 0:
                dset = dset.prefetch(((prefetch_mb << 20) - 1) // bytes_per_item + 1)
            dset = dset.batch(self._tf_minibatch_in)
            self._tf_dataset = dset

            self._tf_iterator = tf.data.Iterator.from_structure(
                tf.compat.v1.data.get_output_types(self._tf_dataset),
                tf.compat.v1.data.get_output_shapes(self._tf_dataset),
            )
            self._tf_init_op = self._tf_iterator.make_initializer(self._tf_dataset)

    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size):
        assert minibatch_size >= 1
        if self._cur_minibatch != minibatch_size:
            self._tf_init_op.run({self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self):  # => images, labels
        return self._tf_iterator.get_next()

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size):  # => images, labels
        self.configure(minibatch_size)
        with tf.name_scope('Dataset'):
            if self._tf_minibatch_np is None:
                self._tf_minibatch_np = self.get_minibatch_tf()
            return tflib.run(self._tf_minibatch_np)

    def get_random_labels_tf(self, minibatch_size):  # => labels
        with tf.name_scope('Dataset'):
            if self.label_size > 0:
                if self.from_tfrecords:
                    with tflex.device('/cpu:0'):
                        return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self.num_samples, dtype=tf.int32))
                else:
                    return tf.random.uniform([minibatch_size], maxval=self.label_size, dtype=tf.int32)
            return tf.zeros([minibatch_size], dtype=tf.int32)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size):  # => labels
        if self.label_size > 0:
            if self.from_tfrecords:
                return self._np_labels[np.random.randint(self.num_samples, size=[minibatch_size])]
            else:
                return np.random.randint(self.label_size, size=[minibatch_size])
        return np.zeros([minibatch_size], dtype=tf.int32)

    # Parse individual image from a tfrecords file into TensorFlow expression.
    def parse_tfrecord_tf(self, record, resize=False):
        features = tf.parse_single_example(record, features={
            'shape': tf.FixedLenFeature([3], tf.int64),
            'data': tf.FixedLenFeature([], tf.string)})
        data = tf.decode_raw(features['data'], tf.uint8)
        data = tf.reshape(data, features['shape'])
        if resize:
            image = tf.transpose(data, [1, 2, 0])
            image = tf.image.resize(image, self.shape[1:])
            data = tf.transpose(image, [2, 0, 1])
        data.set_shape(self.shape)
        return data

    def parse_tfdataset_tf(self, record, resize=False):
        image, label = record['image'], tf.cast(record['label'], tf.int32)
        if resize:
            image = tf.image.resize(image, self.shape[1:])
        image = tf.transpose(image, [2, 0, 1])
        return image, label

    # Parse individual image from a tfrecords file into NumPy array.
    @staticmethod
    def parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value  # pylint: disable=no-member
        data = ex.features.feature['data'].bytes_list.value[0]  # pylint: disable=no-member
        return np.fromstring(data, np.uint8).reshape(shape)

# ----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.


def load_dataset(class_name=None, data_dir=None, verbose=False, **kwargs):
    kwargs = dict(kwargs)
    if 'tfrecord_dir' in kwargs:
        if class_name is None:
            class_name = __name__ + '.TFRecordDataset'
        if data_dir is not None:
            kwargs['tfrecord_dir'] = os.path.join(data_dir, kwargs['tfrecord_dir'])

    assert class_name is not None
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset

# ----------------------------------------------------------------------------
