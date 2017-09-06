import re
import os
import sys
import traceback
import shutil
import numpy as np
import tensorflow as tf
from scipy import misc
from glob import glob
from tqdm import tqdm


# # # For prepare tf records
def get_file_list(path, im_ext):
    print 'Getting files from: %s' % path
    files = glob(os.path.join(path, '*' + im_ext))
    print 'Found %s files' % len(files)
    return files


def move_files(files, target_dir):
    for idx in files:
        shutil.copyfile(idx, target_dir + re.split('/', idx)[-1])


def load_image(file):
    im = misc.imread(file)
    if len(im.shape) < 3:
        im = np.repeat(im[:, :, None], 3, axis=-1)
    if im.shape[-1] != 3:
        im = np.repeat(im[:, :, 0], 3, axis=-1)
    return im


def find_label(files):
    _, c = np.unique(
        [re.split('/', l)[-2] for l in files], return_inverse=True)
    return c


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def image_converter(im_ext):
    if im_ext == '.jpg' or im_ext == '.jpeg' or im_ext == '.JPEG':
        out_fun = tf.image.encode_jpeg
    elif im_ext == '.png':
        out_fun = tf.image.encode_png
    else:
        print '-'*60
        traceback.print_exc(file=sys.stdout)
        print '-'*60
    return out_fun


def create_heatmap(
        obj,
        im_size,
        click_box_radius,
        scoring='linear_decrease',  # lambda x, y: np.add(x, y) lambda x, y: np.maximum(x, y)
        combine_function=lambda x, y: np.add(x, y)):
    canvas = np.zeros((im_size[:2]))
    # xs = np.asarray([int(np.round(float(x))) for x in obj['x']])
    # ys = np.asarray([int(np.round(float(y))) for y in obj['y']])
    xs = np.asarray(obj['x']).astype(np.float).round()
    ys = np.asarray(obj['y']).astype(np.float).round()
    nan_xs = np.where(np.isnan(xs))[0]
    nan_ys = np.where(np.isnan(ys))[0]
    if len(nan_xs) > 0:
        rep_vals = xs[nan_xs - 1]  # replace w/ preceeding value
        xs[nan_xs] = rep_vals
    if len(nan_ys) > 0:
        rep_vals = ys[nan_ys - 1]
        ys[nan_ys] = rep_vals
    xs = xs.astype(int)
    ys = ys.astype(int)

    # Adjust heatmap scoring
    if scoring is 'uniform':
        score = np.ones((len(xs)))
    elif scoring is 'linear_decrease':
        score = np.linspace(0.5, 1., len(xs))[::-1]
    elif scoring is 'linear_increase':
        score = np.linspace(0.5, 1., len(xs))

    for idx in range(len(xs)):  # transpose clicks for js -> python
        # accumulate signal at this spot
        canvas[
            ys[idx] - click_box_radius: ys[idx] + click_box_radius,
            xs[idx] - click_box_radius: xs[idx] + click_box_radius] = \
            combine_function(canvas[
                ys[idx] - click_box_radius: ys[idx] + click_box_radius,
                xs[idx] - click_box_radius: xs[idx] + click_box_radius],
            np.zeros_like(canvas[
                ys[idx] - click_box_radius: ys[idx] + click_box_radius,
                xs[idx] - click_box_radius: xs[idx] + click_box_radius]
                ) + score[idx]
            )
    return canvas


def heatmap_example(image, label, heatmap=None, click_count=None):
    # im = (image / np.max(image)).tostring()
    im = image.tostring()
    if heatmap is False:
        # Training image has an empty clickmap. Toss it.
        # print 'Found an empty clickmap, excluding.'
        return None
    elif heatmap is None:
        heatmap = np.asarray([])
    hm = heatmap.tostring()

    # Set click_count to 0 for images where we don't have maps
    if click_count is None:
        click_count = 0

    # construct the Example proto boject
    return tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
            # Features has a map of string to Feature proto objects
            feature={
                # A Feature contains one of either a int64_list,
                # float_list, or bytes_list
                'label': int64_feature(label),
                'image': bytes_feature(im),
                'heatmap': bytes_feature(hm),
                'click_count': int64_feature(click_count)
            }
        )
    )


def extract_to_tf_records(
        files,
        label_list,
        output_pointer,
        config,
        heatmaps=None,
        click_counts=None):
    print 'Building: %s' % output_pointer
    image_count = 0
    with tf.python_io.TFRecordWriter(output_pointer) as tfrecord_writer:
        for idx, (f, l) in tqdm(
            enumerate(
                zip(files, label_list)), total=len(files)):
            image = load_image(f).astype(np.float32)
            if not all(np.asarray(image.shape) != config.image_size):
                image = misc.imresize(
                    image, config.image_size[:2]).astype(np.float32)
            if heatmaps is not None:  # If we want to encode a heatmap
                if click_counts[idx] > 0:
                    it_hm = create_heatmap(
                        heatmaps[idx],
                        config.image_size,
                        config.click_box_radius,
                        config.hm_scoring).astype(np.float32)
                    prop_check = np.sum(it_hm > 0).astype(float) / it_hm.size
                    var_check = np.var(it_hm) != 0
                    if np.logical_and(  # There is variation
                        np.logical_and(
                            prop_check < 0.99,  # < 99% has entries
                            prop_check > 0.01  # > 1% has entries
                            ), var_check):
                        example = heatmap_example(
                            image,
                            l,
                            heatmap=it_hm,
                            click_count=click_counts[idx])
                    else:
                        example = None
                else:
                    # These are the ILSVRC12 images we don't have clickmaps for
                    example = heatmap_example(
                        image,
                        l,
                        heatmap=np.zeros(
                            (config.image_size[:2]),
                            dtype=np.float32),
                        click_count=click_counts[idx])
            else:
                example = heatmap_example(
                    image, l, heatmap=None)

            if example is not None:
                # Keep track of how many images we use
                image_count += 1
                # use the proto object to serialize the example to a string
                serialized = example.SerializeToString()
                # write the serialized object to disk
                tfrecord_writer.write(serialized)
                example = None
    print 'Finished %s with %s images (dropped %s)' % (
        output_pointer, image_count, len(label_list) - image_count)
