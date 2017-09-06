import re
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from glob import glob
from scipy.ndimage.filters import gaussian_filter


def derive_label(image_filenames, in_dict):
    labels = []
    for im in image_filenames:
        for k, v in in_dict.iteritems():
            label_name = re.split('\d+', re.split('/', im)[-1])[0]
            if label_name == k:
                labels += [v]
    return labels


def make_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


def fine_tune_prepare_layers(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v)
        else:
            other_vars.append(v)
    return other_vars, ft_vars


def fine_tune_names(tf_vars, finetune_vars):
    ft_vars = []
    other_vars = []
    for v in tf_vars:
        ss = [v.name.find(x) != -1 for x in finetune_vars]
        if True in ss:
            ft_vars.append(v.name)
        else:
            other_vars.append(v.name)
    return other_vars, ft_vars


def resize(heatmap, target_tensor):
    target_shape = target_tensor.get_shape()
    return tf.image.resize_bilinear(heatmap, target_shape[1:3])


def list_intersection(a, b):
    return list(set(a) & set(b))


def get_node_tensors(model, attention_layers):
        return [
            v for k, v in model.var_dict.iteritems()
            if k[0] in attention_layers]


def filter_node_names(names):
    # matches = [re.match('(?<=\/)(\w+)(?=\/)', x) for x in names]
    # return [x.group() for x in matches if x is not None]
    return [str(re.split('/', x)[1]) for x in names]


def node_names(cat_loss):
    out = []
    for x in cat_loss:
        if x[0] is not None:
            out.append(re.split('/', x[0].name)[3])
        else:
            out.append('input')
    return out


def find_ckpts(config, dirs=None):
    if dirs is None:
        dirs = sorted(
            glob(
                config.train_checkpoint + config.which_dataset + '*'),
            reverse=True)[0]  # only the newest model run
    ckpts = sorted(glob(dirs + '/*.ckpt*'))
    # Do not include meta files
    ckpts = [ck for ck in ckpts if '.meta' not in ck]
    ckpt_names = [re.split('-', ck)[-1] for ck in ckpts]
    return np.asarray(ckpts), np.asarray(ckpt_names)


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


def eval_top_5(scores, y):
    correct = []
    for sc, it_y in zip(scores, y):
        sorted_sc = np.argsort(sc)[::-1][:5]
        correct.append(1 if it_y in sorted_sc else 0)
    return correct


def plot_image_grad(
    ims, im_names, ops,
        title=None, cmps=None, ro=2, co=2, vmin=-3, vmax=3):
    from matplotlib import pyplot as plt
    count = 0
    if len(vmin) == 1:
        vmin = np.repeat(vmin, len(ims))
    if len(vmax) == 1:
        vmax = np.repeat(vmax, len(ims))
    fig, axes = plt.subplots(nrows=ro, ncols=co, figsize=(24, 10))
    for imset in ims:
        for idx, im in enumerate(imset):
            ax = axes.flat[count]
            out_im = ax.imshow(
                ops[count](im), cmap=cmps[count],
                vmin=vmin[count], vmax=vmax[count])
            ax.tick_params(axis=u'both', which=u'both', length=0)
            ax.set_title(im_names[count])
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            count += 1
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(out_im, cax=cbar_ax)
    plt.title(title)
    plt.savefig('im_grid.png')
    # plt.show()


def gauss_filter(size, sigma, ndims=1):
    # make sure size is odd
    x = int(size)
    if x % 2 == 0:
        x = x + 1
    x_zeros = np.zeros((size, size))

    center = int(np.floor(x / 2.))

    x_zeros[center, center] = 1
    y = gaussian_filter(
        x_zeros, sigma=sigma)[:, :, None, None]
    y = np.repeat(y, ndims, axis=2)
    return y


def blur(image, kernel=3, sigma=7):
    # im_name = re.search('(?<=\/).+(?=\:)', image.name).group()
    with tf.variable_scope('blur'):
        try:
            k = tf.get_variable('GaussKernel%d%d' % (kernel, sigma),
                initializer=gauss_filter(
                kernel, sigma, ndims=int(image.get_shape()[-1])).astype(
                np.float32), dtype=tf.float32, trainable=False)
        except ValueError:
            scope.reuse_variables()
            k = tf.get_variable('GaussKernel%d%d' % (kernel, sigma),
                initializer=gauss_filter(
                kernel, sigma, ndims=int(image.get_shape()[-1])).astype(
                np.float32), dtype=tf.float32, trainable=False)
    return tf.nn.conv2d(image, k, [1, 1, 1, 1], padding='SAME')


def pdist(x, y):
    """ Compute Pairwise (Squared Euclidean) Distance
    Input:
        x: embedding of size M x D
        y: embedding of size N x D
    Output:
        dist: pairwise distance of size M x N
    """

    x2 = tf.tile(tf.expand_dims(tf.reduce_sum(tf.square(x), 1), 1),
                 tf.stack([1, tf.shape(y)[0]]))
    y2 = tf.tile(tf.transpose(tf.expand_dims(tf.reduce_sum(
        tf.square(y), 1), 1)), tf.stack([tf.shape(x)[0], 1]))
    xy = tf.matmul(x, y, transpose_b=True)
    return x2 - 2 * xy + y2


def assign_label(label, x, cluster_center):
    """ Assign Labels
    Input:
        x: embedding of size N x D
        label: cluster label of size N X 1
        K: number of clusters
        tf_eps: small constant
    Output:
        cluster_center: cluster center of size K x D
    """

    dist = pdist(x, cluster_center)
    return label.assign(tf.argmin(dist, 1))


def check_early_stop(
        perf_history,
        minimum_length=20,
        short_history=3,
        long_history=5,
        fail_function=np.less_equal):
    """
    Determine whether to stop early. Using deepgaze criteria:

    We determine this point by comparing the performance from
    the last three epochs to the performance five epochs before those.
    Training runs for at least 20 epochs, and is terminated if all three
    of the last epochs show decreased performance or if
    800 epochs are reached.

    """
    if len(perf_history) < minimum_length:
        early_stop = False
    else:
        short_perf = perf_history[-short_history:]
        long_perf = perf_history[-long_history + short_history:short_history]
        short_check = fail_function(np.mean(long_perf), short_perf)
        if all(short_check):  # If we should stop
            early_stop = True
        else:
            early_stop = False

    return early_stop


def training_loop(
        config,
        coord,
        sess,
        train_op,
        step_op,
        summary_op,
        summary_writer,
        loss,
        saver,
        threads,
        out_dir,
        summary_dir,
        validation_data,
        val_accuracy,
        train_accuracy,
        lr,
        map_loss=None,
        class_loss=None):

    losses, time_elapsed = [], 0
    all_vals = []
    val_accs = np.zeros((config.top_n_validation))
    try:
        while not coord.should_stop():
            start_time = time.time()
            _, loss_value, train_acc, step, lr_value = sess.run(
                [train_op,
                loss,
                train_accuracy,step_op,lr])
            losses.append(loss_value)
            step = step - 1
            duration = time.time() - start_time
            assert not np.isnan(loss_value).any(), 'Model diverged with loss = NaN'
            if step % config.validation_iters == 0:
                if validation_data is not False:
                    it_val_acc = np.asarray([])
                    for num_vals in range(config.num_validation_evals):
                        # Validation accuracy as the average of n batches
                        it_val_acc = np.append(
                            it_val_acc, sess.run(val_accuracy))
                    val_acc = it_val_acc.mean()
                    all_vals += [val_acc]
                else:
                    all_vals = [1.]
                    val_acc -= 1.  # Store every checkpoint

                # Summaries
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

                # Training status and validation accuracy
                format_str = (
                    '%s: step %d, loss = %.2f (%.1f examples/sec; '
                    '%.3f sec/batch) | Training accuracy = %s | '
                    'Validation accuracy = %s | logdir = %s')
                print (format_str % (
                    datetime.now(), step, loss_value,
                    config.train_batch / duration, float(duration),
                    train_acc, val_acc, summary_dir))
                print "current learning rate: ", lr_value

                # Save the model checkpoint if it's the best yet
                if config.top_n_validation > 0:
                    rep_idx = val_acc > val_accs
                    if sum(rep_idx) > 0:
                        force_save = True
                        val_accs[np.argmax(rep_idx)] = val_acc
                else:
                    force_save = True

                if force_save:
                    ckpt_path = os.path.join(
                            config.train_checkpoint,
                            'model_' + str(step) + '.ckpt')
                    saver.save(
                        sess, ckpt_path, global_step=step)
                    print 'Saved checkpoint to: %s' % ckpt_path
                    force_save = False

                    time_elapsed += float(duration)
        
                early_stop = check_early_stop(all_vals)
                if early_stop:
                    'Triggered an early stop.'
                    break
            else:
                # Training status
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; '
                              '%.3f sec/batch) | Training accuracy = %s')
                print (format_str % (datetime.now(), step, loss_value,
                                     config.train_batch / duration,
                                     float(duration), train_acc))

            # End iteration
            step += 1

    except tf.errors.OutOfRangeError:
        print 'Done training for %d epochs, %d steps.' % (config.epochs, step)
        print 'Saved to: %s' % config.train_checkpoint
    finally:
        coord.request_stop()
        np.save(os.path.join(out_dir, 'training_loss'), losses)
    coord.join(threads)
    sess.close()
