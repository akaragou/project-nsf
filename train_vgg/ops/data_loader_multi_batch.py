import re
import tensorflow as tf
import numpy as np
from scipy import misc


def read_and_proc_images(image_names):
    """Depreciated."""
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


def get_crop_coors(image_size, target_size, batch_size, normalize=True):
    """Derive random crop coordinates."""
    h_diff = image_size[0] - target_size[0]
    w_diff = image_size[1] - target_size[1]
    ts_h = tf.constant(
        [target_size[0]] * batch_size, shape=[batch_size])
    ts_w = tf.constant(
        [target_size[1]] * batch_size, shape=[batch_size])
    offset_h = tf.cast(
        tf.round(tf.random_uniform([batch_size], minval=0, maxval=h_diff)),
        tf.int32)
    offset_w = tf.cast(
        tf.round(tf.random_uniform([batch_size], minval=0, maxval=w_diff)),
        tf.int32)
    if normalize:
        offset_h = tf.expand_dims(
            (tf.cast(offset_h, tf.float32)) / tf.cast(
                target_size[0], tf.float32), axis=1)
        ts_h = tf.expand_dims(
            (tf.cast(ts_h, tf.float32)) / tf.cast(
                target_size[0], tf.float32), axis=1)
        offset_w = tf.expand_dims(
            (tf.cast(offset_w, tf.float32)) / tf.cast(
                target_size[1], tf.float32), axis=1)
        ts_w = tf.expand_dims(
            (tf.cast(ts_w, tf.float32)) / tf.cast(
                target_size[1], tf.float32), axis=1)
    return offset_h, ts_h, offset_w, ts_w


def slice_op(image_slice, h_min, w_min, h_max, w_max):
    """Crop image."""
    return tf.slice(
        image_slice, tf.cast(
            tf.concat(axis=0, values=[h_min, w_min]), tf.int32), tf.cast(
            tf.concat(axis=0, values=[h_max, w_max]), tf.int32))


def apply_crop(image, target, h_min, w_min, h_max, w_max):
    """Apply crop from specified coordinates."""
    im_size = image.get_shape()
    if len(im_size) > 2:  # Handle tensors different than matrices
        channels = []
        if len(im_size) == 4:
            image = tf.squeeze(image)  # 4D tensor
            expand_dims = True
        else:
            expand_dims = False
        for idx in range(int(im_size[-1])):
            channels.append(
                slice_op(image[:, :, idx], h_min, w_min, h_max, w_max))
        out_im = tf.stack(channels, axis=2)
        out_im.set_shape([target[0], target[1], int(im_size[-1])])
        if expand_dims:
            out_im = tf.expand_dims(out_im, axis=0)  # Expand back to 4D
        return out_im
    else:  # Matrix route
        out_im = slice_op(image, h_min, w_min, h_max, w_max)
        return out_im.set_shape([target[0], target[1]])


def random_crop(image, heatmap, im_size, model_input_shape, return_heatmaps):
    """Deriving random crop coordinates and applying them to image/heatmap."""
    batch_size = int(image.get_shape()[0])
    h_min, h_max, w_min, w_max = get_crop_coors(
        image_size=im_size,
        target_size=model_input_shape,
        batch_size=int(image.get_shape()[0]),
        normalize=True)
    boxes = tf.concat([w_min, h_min, w_max, h_max], axis=1)
    box_sizes = tf.constant(model_input_shape)
    box_ind = tf.range(batch_size)
    image = tf.image.crop_and_resize(
        image=image,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=box_sizes)
    if return_heatmaps:
        heatmap = tf.image.crop_and_resize(
            image=heatmap,
            boxes=boxes,
            box_ind=box_ind,
            crop_size=box_sizes)
        return image, heatmap
    else:
        return image


def resize_im_data(image, heatmap, model_input_shape, return_heatmaps):
    """Wrapper to resize both image and heatmap."""
    im = resize(image, model_input_shape)
    if return_heatmaps:
        hm = resize(heatmap, model_input_shape)
        return im, hm
    else:
        return im


def flip_image(image, flip_vector, direction):
    """Wrapper to flip 4D image tensors."""
    image_shape = [int(x) for x in image.get_shape()]
    h_norm = np.asarray(image_shape[0] - 1).astype(np.float32)
    w_norm = np.asarray(image_shape[1] - 1).astype(np.float32)
    flip_vector = tf.expand_dims(
            tf.where(
                tf.equal(flip_vector, False),
                tf.constant([-1] * image_shape[0], dtype=tf.float32),
                tf.constant([1] * image_shape[0], dtype=tf.float32)),
            axis=-1)
    y1 = tf.constant(
        [[0]] * image_shape[0], dtype=tf.float32) / h_norm
    y2 = tf.constant(
        [[image_shape[0]]] * image_shape[0], dtype=tf.float32) / h_norm
    x1 = tf.constant(
        [[0]] * image_shape[0], dtype=tf.float32) / w_norm
    x2 = tf.constant(
        [[image_shape[1]]] * image_shape[0], dtype=tf.float32) / w_norm
    if direction == 'lr':
        old_x1 = tf.identity(x1)
        old_x2 = tf.identity(x2)
        x1 = tf.where(
                tf.equal(flip_vector, True),
                old_x1,
                old_x2)
        x2 = tf.where(
                tf.equal(flip_vector, True),
                old_x2,
                old_x1)
    elif direction == 'ud':
        old_y1 = tf.identity(y1)
        old_y2 = tf.identity(y2)
        y1 = tf.where(
                tf.equal(flip_vector, True),
                old_y1,
                old_y2)
        y2 = tf.where(
                tf.equal(flip_vector, True),
                old_y2,
                old_y1)
    else:
        raise RuntimeError('Could not understand the flip direction.')
    boxes = tf.concat([y1, x1, y2, x2], axis=1)
    box_ind = tf.range(image_shape[0])
    crop_size = tf.constant(image_shape[1:3])
    image = tf.image.crop_and_resize(
        image=image,
        boxes=boxes,
        box_ind=box_ind,
        crop_size=crop_size)
    return image


def image_augmentations(
        image, heatmap, im_size, data_augmentations,
        model_input_shape, return_heatmaps):
    """Coordinating image augmentations for both image and heatmap."""
    if data_augmentations is not None:
        if 'random_crop' in data_augmentations:
            im_data = random_crop(
                image,
                heatmap,
                im_size,
                model_input_shape,
                return_heatmaps)
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
            batch_size = int(image.get_shape()[0])
            lorr = tf.less(
                tf.random_uniform(
                    [batch_size],
                    minval=0,
                    maxval=1.),
                .5)
            image = flip_image(
                image=image,
                flip_vector=lorr,
                direction='lr')
            if return_heatmaps:
                heatmap = flip_image(
                    image=heatmap,
                    flip_vector=lorr,
                    direction='lr')
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
        heatmap = tf.expand_dims(heatmap[:, :, :, 0], axis=-1)
        return image, heatmap
    else:
        return image


def resize(image, target_shape):
    """Wrapper for image resizing."""
    return tf.image.resize_bilinear(image, target_shape)


def get_tf_dict(return_heatmaps, return_clicks):
    """Prepare feature dictionaries for tfrecord decoding."""
    fdict = {
              'label': tf.FixedLenFeature([], tf.int64),
              'image': tf.FixedLenFeature([], tf.string)
            }

    if return_heatmaps:
        fdict['heatmap'] = tf.FixedLenFeature([], tf.string)
    if return_clicks:
        fdict['click_count'] = tf.FixedLenFeature([], tf.int64)
    return fdict


def read_and_decode(
        filename_queue,
        im_size,
        model_input_shape,
        data_augmentations,
        return_heatmaps,
        return_clicks,
        batch_size):
    """Read and decode tensors from tf_records and apply augmentations."""
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read_up_to(filename_queue, batch_size)
    fdict = get_tf_dict(
        return_heatmaps=return_heatmaps,
        return_clicks=return_clicks)
    features = tf.parse_example(serialized_example, features=fdict)

    # Handle decoding of each element
    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.cast(features['label'], tf.int32)
    if return_heatmaps:
        heatmap = tf.decode_raw(features['heatmap'], tf.float32)
    else:
        heatmap = None

    if return_clicks:
        counts = tf.cast(features['click_count'], tf.float32)
    else:
        counts = None

    # Reshape each element
    image = tf.reshape(image, [batch_size] + im_size)
    label = tf.reshape(label, [batch_size])
    if heatmap is not None:
        heatmap = tf.reshape(heatmap, [batch_size, im_size[0], im_size[1], 1])
    if counts is not None:
        counts = tf.reshape(counts, [batch_size, 1])

    # Preprocess images and heatmaps
    if heatmap is not None:
        heatmap /= tf.reduce_max(
            heatmap,
            reduction_indices=[1, 2],
            keep_dims=True)
        heatmap = repeat_elements(heatmap, 3, axis=3)
        image, heatmap = image_augmentations(
            image,
            heatmap,
            im_size,
            data_augmentations,
            model_input_shape,
            return_heatmaps)
        heatmap = tf.where(tf.is_nan(heatmap), tf.zeros_like(heatmap), heatmap)
    else:
        image = image_augmentations(
            image,
            None,
            im_size,
            data_augmentations,
            model_input_shape,
            return_heatmaps)

    if return_heatmaps is True and return_clicks is False:
        return image, label, heatmap
    elif return_heatmaps is True and return_clicks is True:
        return image, label, heatmap, counts
    elif return_heatmaps is False and return_clicks is True:
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
        return_clicks=False):
    """Read tfrecords and prepare them for queueing."""
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
            return_clicks=return_clicks,
            batch_size=batch_size)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.
        if return_heatmaps and return_clicks:
            if shuffle_batch:
                images, labels, heatmaps, counts = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many,
                    allow_smaller_final_batch=False)
            else:
                images, labels, heatmaps, counts = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity)

            return images, labels, heatmaps, counts
        elif return_heatmaps and return_clicks is False:
            if shuffle_batch:
                images, labels, heatmaps = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many,
                    allow_smaller_final_batch=False)
            else:
                images, labels, heatmaps = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    enqueue_many=enqueue_many,
                    allow_smaller_final_batch=False)

            return images, labels, heatmaps
        else:
            if shuffle_batch:
                images, labels = tf.train.shuffle_batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    # Ensures a minimum amount of shuffling of examples.
                    min_after_dequeue=min_after_dequeue,
                    enqueue_many=enqueue_many,
                    allow_smaller_final_batch=False)
            else:
                images, labels = tf.train.batch(
                    batch_data, batch_size=batch_size, num_threads=num_threads,
                    capacity=capacity,
                    enqueue_many=enqueue_many,
                    allow_smaller_final_batch=False)

            return images, labels
