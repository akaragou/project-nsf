import os
import sys
import tensorflow as tf
from ops import data_loader
from config import clickMeConfig
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from tensorflow.python.client import timeline


def run_tester(config, cv, plot_images=False, break_point=1000):
    if cv == 'train':
        data = os.path.join(
            config.tf_record_base,
            'cast_clicks_only_train_7.tfrecords')
        return_heatmaps = True
    elif cv == 'test':
        data = os.path.join(
            config.tf_record_base,
            config.tf_val_name)
        return_heatmaps = False
    else:
        raise RuntimeError('Pass a train/test flag.')
    images, labels, heatmaps = data_loader.inputs(
        data,
        64,
        config.image_size,
        config.model_image_size[:2],
        train=config.data_augmentations,
        num_epochs=1,
        return_heatmaps=return_heatmaps,
        shuffle_batch=False)

    sess = tf.Session()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Required. See below for explanation
    sess.run(
        tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()),
        options=run_options,
        run_metadata=run_metadata)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # first example from file
    count = 0
    try:
        while not coord.should_stop():
            start = timer()
            it_im, it_lab, it_hm = sess.run(
                [images, labels, heatmaps],
                options=run_options,
                run_metadata=run_metadata)
            delta = timer() - start
            sys.stdout.write(
                '\rBatch %s extracted in: %s seconds' % (count, delta)),
            sys.stdout.flush()
            count += 1
            if plot_images:
                f = plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(it_im)
                plt.title(
                    'Image size: %s, %s, %s' % (
                        it_im.shape[0], it_im.shape[1], it_im.shape[2]))
                plt.subplot(1, 2, 2)
                plt.imshow(it_hm[:, :, 0])
                plt.title('Category: %s' % it_lab)
                plt.show()
                plt.close(f)
            if it_lab is None:
                sys.stdout.write('\n')
                break
            if count > break_point:
                break
    except tf.errors.OutOfRangeError:
        print 'Finished.'
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format()
    with open('timeline.json', 'w') as f:
        f.write(ctf)
    print "all done"


if __name__ == '__main__':
    config = clickMeConfig()
    run_tester(config, cv='train')
