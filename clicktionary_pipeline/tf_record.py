#!/usr/bin/env python
from __future__ import division
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
from tqdm import tqdm
from scipy import misc
from random import randint

VGG_MEAN = [103.939, 116.779, 123.68]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value=[value]))

def tfrecord2metafilename(tfrecord_filename):
    """
    Derive associated meta filename for a tfrecord filename
    Input: /path/to/foo.tfrecord
    Output: /path/to/foo.meta
    """
    base, ext = os.path.splitext(tfrecord_filename)
    return base + '_meta.npz'

def vgg_preprocessing(image_rgb):
    """
    Preprocssing the given image for evalutaiton with vgg16 model 
    Input: image_rgb - A tensor representing an image of size [224, 224, 3]
    Output: A processed image 
    """

    image_rgb_scaled = image_rgb * 255.0
    red, green, blue = tf.split(num_or_size_splits=3, axis=3, value=image_rgb_scaled)
    assert red.get_shape().as_list()[1:] == [224, 224, 1]
    assert green.get_shape().as_list()[1:] == [224, 224, 1]
    assert blue.get_shape().as_list()[1:] == [224, 224, 1]
    image_bgr = tf.concat(values = [
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
        ], axis=3)
    assert image_bgr.get_shape().as_list()[1:] == [224, 224, 3], image_bgr.get_shape().as_list()
    return image_bgr


def distort_color(image, color_ordering=0, fast_mode=False, scope=None):
  """
  Source: https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py
  Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def create_tf_record(tfrecords_filename, file_pointers, labels=None, is_grayscale=False, resize=True):
    """
    Creates tf records by writing image data to binary files (allows for faster 
    reading of data). Meta data for the tf records is stored as well.
    Inputs: tfrecords_filename - Directory to store tfrecords
            file_pointers - Empty list of filepointers pointing to the location of the images
            labels - Empty list of labels that correspond to the image file_pointers. 
            If labels remains None tf records will be created without labels (useful for predicting labels on test images)
            is_grayscale - Boolean for whether the images are grayscale or not
    Output: None
    """
    
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    if not labels:

        print '%d files' % (len(np.unique(file_pointers)))
    
        for img_path in tqdm(file_pointers):
           
            img = np.array(Image.open(img_path))

            if is_grayscale:
                img = np.expand_dims(img,-1)
                img = np.repeat(img, 3, -1)

            if resize:
                img = misc.imresize(img, (256, 256)) 
        
            img_raw = img.tostring()
            path_raw = img_path.encode('utf-8')

            example = tf.train.Example(features=tf.train.Features(feature={
        
                    'image_raw': _bytes_feature(img_raw),
                    'file_path': _bytes_feature(path_raw),

                   }))

            writer.write(example.SerializeToString())

        writer.close()

       
    else:

        print '%d files in %d categories' % (len(np.unique(file_pointers)), len(np.unique(labels)))

        for img_path, l in tqdm(zip(file_pointers, labels)):

                img = np.array(Image.open(img_path))

                if is_grayscale:
                    img = np.expand_dims(img,-1)
                    img = np.repeat(img, 3, -1)

                if resize:
                    img = misc.imresize(img, (256, 256)) 
            
                img_raw = img.tostring()
                path_raw = img_path.encode('utf-8')

                example = tf.train.Example(features=tf.train.Features(feature={
            
                        'image_raw': _bytes_feature(img_raw),
                        'file_path': _bytes_feature(path_raw),
                        'label':_int64_feature(int(l)),

                       }))

                writer.write(example.SerializeToString())

        writer.close()

        meta = tfrecord2metafilename(tfrecords_filename)
        np.savez(meta, file_pointers=file_pointers, labels=labels, output_pointer=tfrecords_filename)

    print '-' * 100
    print 'Generated tfrecord at %s' % tfrecords_filename
    print '-' * 100


def read_and_decode(filename_queue=None, img_dims=[256,256,3], resize_to=[256,256], model_dims=[224,224,3], size_of_batch=32,\
                     labels=True, augmentations_dic=None, num_of_threads=1, shuffle=True):

    """
    Reads in tf records and decodes the features of the image 
    Input: filename_queue - A node in a TensorFlow Graph used for asynchronous computations
           img_dims - Dimensions of the tensor image stored as a tfrecord, example: [256, 256, 3] 
           model_dims - Dimensions of the tensor image that the model accepts, example: [224, 224, 3] 
           resize_to - Size to resize tf record to before training if resize_to is the same a img_dims no resizing will take place
           size_of_batch - Size of the batch that will be fed into the model, example: 32
           labels - Option for if the images stored in tfrecords have labels associated with them
           augmentations_dic - Dictionary of augmentations that an image can have for training and validation. Augmentations
           are chosen in the config
           num_threads - Number of threads that execute a training op that dequeues mini-batches from the queue 
           shuffle - Boolean if batches fed into graph should be shuffled or not 
    Outputs: Tensor image, label of the image and filepath to the image. If labels is False only tensor image and filepath will be returned
    """
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    
    if not labels:
        features = tf.parse_single_example(
          serialized_example,
        
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'file_path': tf.FixedLenFeature([], tf.string),
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        file_path = tf.cast(features['file_path'], tf.string)
        
        
        image = tf.reshape(image, img_dims)
        image = tf.cast(image, tf.float32)

        image = tf.image.resize_images(image, resize_to)

        image = tf.to_float(image)
        image = image/255

        if augmentations_dic and augmentations_dic['scale_jitter']:
            random_size = randint(256,512)
            image = tf.image.resize_images(image, [random_size, random_size])
        else:
            image = tf.image.resize_images(image,resize_to)

        if augmentations_dic and  augmentations_dic['rand_crop']:
            image = tf.random_crop(image, model_dims)

        else:
            image = tf.image.resize_image_with_crop_or_pad(image, model_dims[0],\
                                                         model_dims[1])

        if augmentations_dic and  augmentations_dic['rand_color']:
            random_color_ordering = randint(0,3)
            image = distort_color(image,random_color_ordering)

        if augmentations_dic and augmentations_dic['rand_flip_left_right']:
            image = tf.image.random_flip_left_right(image)

        if augmentations_dic and augmentations_dic['rand_flip_top_bottom']:
            image = tf.image.random_flip_up_down(image)

        if augmentations_dic and augmentations_dic['rand_rotate']:
            random_angle = randint(0,359)
            image = tf.contrib.image.rotate(image, random_angle)

        if shuffle:
      
            img, f = tf.train.shuffle_batch([image, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=1000 + 3 * size_of_batch,
                                                         min_after_dequeue=1000,
                                                         num_threads=num_of_threads)
        else:
            img, f = tf.train.batch([image, file_path],
                                                         batch_size=size_of_batch,
                                                          capacity=100000,
                                                          allow_smaller_final_batch=True,
                                                         num_threads=num_of_threads)

        
        return img, f

    else:
        features = tf.parse_single_example(
          serialized_example,
        
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'file_path': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)

        label = tf.cast(features['label'], tf.int32)

        file_path = tf.cast(features['file_path'], tf.string)
        
        
        image = tf.reshape(image, img_dims)
        image = tf.cast(image, tf.float32)

        image = tf.image.resize_images(image, resize_to)

        image = tf.to_float(image)
        image = image/255

        if augmentations_dic and augmentations_dic['scale_jitter']:
            random_size = randint(256,512)
            image = tf.image.resize_images(image, [random_size, random_size])
        else:
            image = tf.image.resize_images(image,resize_to)

        if augmentations_dic and augmentations_dic['rand_crop']:
            image = tf.random_crop(image, model_dims)

        else:
            image = tf.image.resize_image_with_crop_or_pad(image, model_dims[0],\
                                                         model_dims[1])

        if augmentations_dic and augmentations_dic['rand_color']:
            random_color_ordering = randint(0,3)
            image = distort_color(image,random_color_ordering)

        if augmentations_dic and augmentations_dic['rand_flip_left_right']:
            image = tf.image.random_flip_left_right(image)

        if augmentations_dic and augmentations_dic['rand_flip_top_bottom']:
            image = tf.image.random_flip_up_down(image)

        if augmentations_dic and augmentations_dic['rand_rotate']:
            random_angle = randint(0,359)
            image = tf.contrib.image.rotate(image, random_angle)

        if shuffle:
            img, l, f = tf.train.shuffle_batch([image, label, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=1000 + 3 * size_of_batch,
                                                         min_after_dequeue=1000,
                                                         num_threads=num_of_threads)
        else:
            img, l, f = tf.train.batch([image, label, file_path],
                                                         batch_size=size_of_batch,
                                                         capacity=100000,
                                                         allow_smaller_final_batch=True,
                                                         num_threads=num_of_threads)        
        return img, l, f


