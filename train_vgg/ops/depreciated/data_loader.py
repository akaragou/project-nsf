import re
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from scipy import misc


def read_and_proc_images(image_names):
    label = []
    for idx, x in enumerate(image_names):
        im = misc.imread(x)
        if len(im.shape) == 2:
            im = np.repeat(im[:, :, None], 3, axis=-1)
        if im.shape[-1] > 3:
            im = im[:, :, :3]
        if idx == 0:
            images = im[None, :, :, :]
        else:
            images = np.append(images, im[None, :, :, :], axis=0)
        label = np.append(
            label, int(re.search('(?<=\/)(\d+)(?=\_)', x).group()))
    return images, label


def repeat_elements(x, rep, axis):
    '''Repeats the elements of a tensor along an axis, like np.repeat
    If x has shape (s1, s2, s3) and axis=1, the output
    will have shape (s1, s2 * rep, s3)
    This function is taken from keras backend
    '''
    x_shape = x.get_shape().as_list()
    splits = tf.split(axis=axis, num_or_size_splits=x_shape[axis], value=x)
    x_rep = [s for s in splits for i in range(rep)]
    return tf.concat(axis=axis, values=x_rep)


def read_and_decode_single_example(
                    filename, im_size, model_input_shape,
                    data_augmentations,
                    return_heatmaps, weight_loss_with_counts=None):
    """first construct a queue containing a list of filenames.
    this lets a user split up there dataset in multiple files to keep
    size down"""
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    fdict = get_tf_dict(weight_loss_with_counts=weight_loss_with_counts)

    if return_heatmaps:
        fdict['heatmap'] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example([serialized_example], features=fdict)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, im_size)  # np.asarray(im_size)[[2, 0, 1]])

    # image = tf.transpose(res_image, [2, 1, 0])
    image.set_shape(im_size)

    if return_heatmaps:
        # Normalize the heatmap and prep for image augmentations
        heatmap = tf.decode_raw(features['heatmap'], tf.float32)
        heatmap = tf.reshape(heatmap, [im_size[0], im_size[1], 2])
        heatmap = heatmap[:, :, 1]  # This is strange. WHY 2 LAYERS?
        heatmap /= tf.reduce_max(heatmap)
        heatmap = repeat_elements(tf.expand_dims(
            heatmap, 2), 3, axis=2)
        image, heatmap = image_augmentations(
            image, heatmap, im_size, data_augmentations,
            model_input_shape, return_heatmaps)
    else:
        image = image_augmentations(
            image, None, im_size, data_augmentations,
            model_input_shape, return_heatmaps)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    if weight_loss_with_counts:
        counts = tf.cast(features['click_counts'], tf.int32)

    if return_heatmaps:
        return image, label, heatmap
    elif return_heatmaps and weight_loss_with_counts:
        return image, label, heatmap, counts
    elif not return_heatmaps and weight_loss_with_counts:
        return image, label, counts
    else:
        return image, label


def get_crop_coors(image_size, target_size):
    h_diff = image_size[0] - target_size[0]
    ts = tf.constant(
        target_size[0], shape=[2, 1])
    offset = tf.cast(
        tf.round(tf.random_uniform([1], minval=0, maxval=h_diff)), tf.int32)
    return offset, ts[0], offset, ts[1]


def slice_op(image_slice, h_min, w_min, h_max, w_max):
    return tf.slice(
        image_slice, tf.cast(
            tf.concat(axis=0, values=[h_min, w_min]), tf.int32), tf.cast(
            tf.concat(axis=0, values=[h_max, w_max]), tf.int32))


def apply_crop(image, target, h_min, w_min, h_max, w_max):
    im_size = image.get_shape()
    if len(im_size) > 2:
        channels = []
        for idx in range(int(im_size[-1])):
            channels.append(
                slice_op(image[:, :, idx], h_min, w_min, h_max, w_max))
        out_im = tf.stack(channels, axis=2)
        out_im.set_shape([target[0], target[1], int(im_size[-1])])
        return out_im
    else:
        out_im = slice_op(image, h_min, w_min, h_max, w_max)
        return out_im.set_shape([target[0], target[1]])


def random_crop(image, heatmap, im_size, model_input_shape, return_heatmaps):
    h_min, h_max, w_min, w_max = get_crop_coors(
        image_size=im_size, target_size=model_input_shape)
    im = apply_crop(
        image, model_input_shape, h_min, w_min, h_max, w_max)
    if return_heatmaps:
        hm = apply_crop(
            heatmap, model_input_shape, h_min, w_min, h_max, w_max)
        # If the heatmap is empty let's revert to a resize
        hm_test = tf.greater_equal(tf.reduce_sum(hm), 0)
        hm = control_flow_ops.cond(
            hm_test,
            lambda: hm,
            lambda: resize(heatmap, model_input_shape))
        im = control_flow_ops.cond(
            hm_test,
            lambda: im,
            lambda: resize(image, model_input_shape))
        return im, hm
    else:
        return im


def resize_im_data(image, heatmap, model_input_shape, return_heatmaps):
    im = resize(heatmap, model_input_shape)
    if return_heatmaps:
        hm = resize(heatmap, model_input_shape)
        return im, hm
    else:
        return im


def image_augmentations(
        image, heatmap, im_size, data_augmentations,
        model_input_shape, return_heatmaps):

    # Insert augmentation and preprocessing here
    if data_augmentations is not None:
        if 'random_crop' in data_augmentations:
            im_data = random_crop(image, heatmap, im_size, model_input_shape, return_heatmaps) 
            if return_heatmaps:
                image, heatmap = im_data 
            else:
                image = im_data
        elif 'random_crop_resize' in data_augmentations:
            rc = tf.greater(tf.random_uniform([], minval=0, maxval=1), 0.75)  # thresh for rand
            im_data = control_flow_ops.cond(
                rc,
                lambda: resize_im_data(image, heatmap, model_input_shape, return_heatmaps),
                lambda: random_crop(image, heatmap, im_size, model_input_shape, return_heatmaps))
            if return_heatmaps:
                image, heatmap = im_data
            else:
                image = im_data
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, model_input_shape[0], model_input_shape[1])
            if return_heatmaps:
                heatmap = tf.image.resize_image_with_crop_or_pad(
                    heatmap, model_input_shape[0], model_input_shape[1])

        if 'left_right' in data_augmentations:
            lorr = tf.less(tf.random_uniform([], minval=0, maxval=1.), .5)
            image = control_flow_ops.cond(
                lorr,
                lambda: tf.image.flip_left_right(image),
                lambda: image)
            if return_heatmaps:
                heatmap = control_flow_ops.cond(
                    lorr,
                    lambda: tf.image.flip_left_right(heatmap),
                    lambda: heatmap)
        if 'random_contrast' in data_augmentations:
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        if 'random_brightness' in data_augmentations:
            image = tf.image.random_brightness(image, max_delta=63.)

    else:
        image = tf.image.resize_image_with_crop_or_pad(
            image, model_input_shape[0], model_input_shape[1])
        if return_heatmaps:
            heatmap = tf.image.resize_image_with_crop_or_pad(
                heatmap, model_input_shape[0], model_input_shape[1])

    # Make sure to clip values to [0, 1]
    image = image / 255.
    image = tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0)

    if return_heatmaps:
        heatmap = tf.clip_by_value(tf.cast(heatmap, tf.float32), 0.0, 1.0)
        # Only take 1 slice from the heatmap
        heatmap = tf.expand_dims(heatmap[:, :, 0], axis=2)
        return image, heatmap
    else:
        return image


def resize(image, target_shape):
    return tf.squeeze(
        tf.image.resize_bilinear(tf.expand_dims(image, axis=0), target_shape))


def get_tf_dict(return_heatmaps, weight_loss_with_counts):
    fdict = {
              'label': tf.FixedLenFeature([], tf.int64),
              'image': tf.FixedLenFeature([], tf.string)
            }

    if return_heatmaps:
        fdict['heatmap'] = tf.FixedLenFeature([], tf.string)
    if weight_loss_with_counts:
        fdict['click_count'] = tf.FixedLenFeature([], tf.int64)
    return fdict


def read_and_decode(
        filename_queue,
        im_size,
        model_input_shape,
        data_augmentations,
        return_heatmaps,
        weight_loss_with_counts,
        enqueue_many):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    fdict = get_tf_dict(
        return_heatmaps=return_heatmaps,
        weight_loss_with_counts=weight_loss_with_counts)

    features = tf.parse_single_example(serialized_example, features=fdict)

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image'], tf.float32)

    # Need to reconstruct channels first then transpose channels
    image = tf.reshape(image, im_size)  # np.asarray(im_size)[[2, 0, 1]])

    # image = tf.transpose(res_image, [2, 1, 0])
    image.set_shape(im_size)

    if return_heatmaps:
        # Normalize the heatmap and prep for image augmentations
        heatmap = tf.decode_raw(features['heatmap'], tf.float32)
        heatmap = tf.reshape(heatmap, [im_size[0], im_size[1], 2])
        heatmap = heatmap[:, :, 1]  # This is strange. WHY 2 LAYERS?
        heatmap /= tf.reduce_max(heatmap)
        heatmap = repeat_elements(tf.expand_dims(
            heatmap, 2), 3, axis=2)
        image, heatmap = image_augmentations(
            image, heatmap, im_size, data_augmentations,
            model_input_shape, return_heatmaps)
        heatmap = tf.where(tf.is_nan(heatmap), tf.zeros_like(heatmap), heatmap)

    else:
        image = image_augmentations(
            image, None, im_size, data_augmentations,
            model_input_shape, return_heatmaps)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)
    if enqueue_many:
        image = tf.expand_dims(image, axis=0)
        label = tf.expand_dims(label, axis=0)
        if return_heatmaps is True:
            heatmap = tf.expand_dims(heatmap, axis=0)

    if weight_loss_with_counts:
        counts = tf.cast(features['click_count'], tf.float32)

    if return_heatmaps is True and weight_loss_with_counts is False:
        return image, label, heatmap
    elif return_heatmaps is True and weight_loss_with_counts is True:
        return image, label, heatmap, counts
    elif return_heatmaps is False and weight_loss_with_counts is True:
        return image, label, counts
    else:
        return image, label


def inputs(
        tfrecord_file,
        batch_size,
        im_size,
        model_input_shape,
        train=None,
        num_epochs=None,
        num_threads=2,
        return_heatmaps=True,
        shuffle_batch=True,
        weight_loss_with_counts=False):
    min_after_dequeue = batch_size * 5
    capacity = min_after_dequeue + 5 * batch_size
    enqueue_many = True
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [tfrecord_file], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        batch_data = read_and_decode(
            filename_queue=filename_queue,
            im_size=im_size,
            model_input_shape=model_input_shape,
            data_augmentations=train,
            return_heatmaps=return_heatmaps,
            weight_loss_with_counts=weight_loss_with_counts,
            enqueue_many=enqueue_many)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if return_heatmaps and weight_loss_with_counts:
            if shuffle_batch:
                images, labels, heatmaps, counts = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many)
            else:
                images, labels, heatmaps, counts = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity)

            return images, labels, heatmaps, counts
        elif return_heatmaps and weight_loss_with_counts is False:
            if shuffle_batch:
                images, labels, heatmaps = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many)
            else:
                images, labels, heatmaps = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    enqueue_many=enqueue_many)

            return images, labels, heatmaps
        else:
            if shuffle_batch:
                images, labels = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many)
            else:
                images, labels = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    enqueue_many=enqueue_many)

            return images, labels
