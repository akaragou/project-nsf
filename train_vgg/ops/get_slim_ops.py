from __future__ import print_function, division
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import tensorflow.contrib.slim.nets
import models.slim_inception_v3 as inception
from ops.data_loader import inputs
from ops.tf_fun import make_dir, blur
from datetime import datetime
from ops.loss_utils import inception_finetune_learning,\
    combine, channel_l2_normalize
from ops.metrics import class_accuracy
import os.path


def inception_input_processing(images):
    '''Rescale image data from [0, 1] to [-1, 1].'''
    return 2.0 * (images - 0.5)


def get_maps(train_logits, train_labels, train_images, train_heatmaps, cmcfg):
    # Model attention maps
    attention_neurons = train_logits
    if cmcfg.targeted_gradient:
        # Mask so only this class' neuron and its inputs are affected
        attention_neurons *= tf.one_hot(train_labels, cmcfg.output_shape)
    attention_maps = tf.gradients(attention_neurons, train_images)[0]

    # Combine the maps
    comb_attention_maps = combine(attention_maps, cmcfg.combine_type)

    # Blur if nec. Also blur the target maps.
    norm_hms = train_heatmaps
    if cmcfg.blur_maps > 0:
        comb_attention_maps = blur(comb_attention_maps, kernel=cmcfg.blur_maps * 2)
        norm_hms = blur(norm_hms, kernel=cmcfg.blur_maps)

    return comb_attention_maps, norm_hms


def add_attention_gradient_loss(loss, train_logits, train_labels, train_images, train_heatmaps, cmcfg):
    comb_attention_maps, norm_hms = get_maps(train_logits, train_labels, train_images, train_heatmaps, cmcfg)

    # Read config to normalize and compute loss
    if cmcfg.loss_function == 'l2':
        print('L2 attention grad loss. Using l2 normalization')
        comb_attention_maps = channel_l2_normalize(comb_attention_maps)
        norm_hms = channel_l2_normalize(norm_hms)
        map_loss = tf.nn.l2_loss(comb_attention_maps - norm_hms)
    elif cmcfg.loss_function == 'masked_l2':
        print('Masked L2 attention grad loss. Using l2 normalization')
        comb_attention_maps = channel_l2_normalize(comb_attention_maps)
        norm_hms = channel_l2_normalize(norm_hms)
        hm_mask = tf.cast(
            tf.greater(
                norm_hms, tf.reduce_min(norm_hms, reduction_indices=[1, 2, 3])), tf.float32)
        map_loss = tf.nn.l2_loss((hm_mask * comb_attention_maps) - (hm_mask * norm_hms))
    elif cmcfg.loss_function == 'l1':
        print('L1 attention grad loss. Using min-max normalization')
        comb_attention_maps = min_max(comb_attention_maps)
        norm_hms = min_max(train_heatmaps)
        map_loss = tf.reduce_sum(tf.abs(comb_attention_maps - norm_hms))
    elif cmcfg.loss_function == 'log_loss':
        if cmcfg.normalize == 'softmax':
            exp_comb_attention_maps = tf.exp(comb_attention_maps)
            comb_attention_maps = exp_comb_attention_maps / tf.reduce_sum(
                exp_comb_attention_maps, reduction_indices=[1, 2], keep_dims=True)
            exp_hms = tf.exp(train_heatmaps)
            norm_hms = exp_hms / tf.reduce_sum(
                exp_hms, reduction_indices=[1, 2], keep_dims=True)
            map_loss = tf.reduce_mean(tf.losses.log_loss(
                labels=norm_hms,
                predictions=comb_attention_maps))
        elif cmcfg.normalize == 'sigmoid':
            map_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=norm_hms,
                logits=comb_attention_maps))
        else:
            raise RuntimeError('Set cmcfg.normalize to softmax or sigmoid.')

    if cmcfg.loss_type == 'joint':
        return loss + cmcfg.reg_penalty * map_loss
    elif cmcfg.loss_type == 'map':
        return map_loss
    elif cmcfg.loss_type == 'classification':
        return loss
    else:
        raise RuntimeError('Choose a loss type in the config.')


def init_inception_v3(train_dir, validation_dir, cmcfg, incfg,
                      use_attention_gradients, use_modulation,
                      hp_optim, training=True, reuse=None,
                      scope=None):
    '''init_inception_v3

    Get the ops that let you train a slim inceptionv3 using this repo's
    `ops.data_loader.inputs(...)` pipeline, with augmentations to make
    sure data is right for the Inception network.

    Args:
        train_dir: string, folder where training tfrecords and meta stuff live
        validation_dir: string, folder where validation tfrecords and meta live
        cmcfg: a `clickMeConfig` from `config.py`
        incfg: an `InceptionConfig` from `config.py`
        use_attention_gradients: add attn grad to the loss and training ops
        use_modulation: multiply input by normalized heatmap. don't use this
            at the same time as attention gradients
        hp_optim: turn this into a hyperparameter optimization worker
    Returns:

    '''
    assert not (use_attention_gradients and use_modulation)
    # Hyperparameter stuff?
    if hp_optim:
        raise ValueError('Still figuring out HP optimization with InceptionV3.')

    # Figure out data situation
    if train_dir is None:  # Use globals
        train_data = os.path.join(cmcfg.tf_record_base, cmcfg.tf_train_name)
        train_meta_name = os.path.join(
            cmcfg.tf_record_base,
            os.path.splitext(cmcfg.tf_train_name)[0] + '_meta.npz')
        train_meta = np.load(train_meta_name)
    else:
        validation_data = os.path.join(
            validation_dir, cmcfg.tf_val_name)
        val_meta_name = os.path.join(
            validation_dir,
            os.path.splitext(cmcfg.tf_val_name)[0] + '_meta.npz')
        val_meta = np.load(val_meta_name)
    print('Using train tfrecords: %s | %s image/heatmap combos' % 
          ([train_data], len(train_meta['labels'])))

    if validation_dir is True:  # Use globals
        validation_data = os.path.join(
            cmcfg.tf_record_base, cmcfg.tf_val_name)
        val_meta_name = os.path.join(
            cmcfg.tf_record_base,
            os.path.splitext(cmcfg.tf_val_name)[0] + '_meta.npz')
        val_meta = np.load(val_meta_name)
        print('Using validation tfrecords: %s | %s images' %
              (validation_data, len(val_meta['labels'])))
    elif validation_dir is False:
        print('Not using validation data.')
    else:
        validation_data = os.path.join(
            validation_dir, cmcfg.tf_val_name)
        val_meta_name = os.path.join(
            validation_dir,
            os.path.splitext(cmcfg.tf_val_name)[0] + '_meta.npz')
        val_meta = np.load(val_meta_name)
        print('Using validation tfrecords: %s | %s images' %
              (validation_data, len(val_meta['labels'])))

    # Make and timestamp output directories
    dt_stamp = '%s_%s_%d_%s' % \
                ('attngrad' if use_attention_gradients else 'baseline',
                 cmcfg.new_lr, len(train_meta['labels']),
                 datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    cmcfg.train_checkpoint = os.path.join(cmcfg.train_checkpoint, dt_stamp)
    out_dir = os.path.join(cmcfg.results, dt_stamp)
    for d in [cmcfg.train_checkpoint, cmcfg.train_summaries, cmcfg.results, out_dir]:
        make_dir(d) 

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        # Basic data loading
        train_images, train_labels, train_heatmaps = inputs(train_data, cmcfg.train_batch,
                                                            cmcfg.image_size,
                                                            cmcfg.model_image_size[:2],
                                                            train=cmcfg.data_augmentations,
                                                            num_epochs=cmcfg.epochs,
                                                            return_heatmaps=True)

        # Do inception-specific processing to mimic
        # https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py
        # Notably, they rescale to [-1, 1].
        train_images = inception_input_processing(train_images)
        if validation_dir is not False:
            val_images, val_labels = inputs(validation_data,
                                            cmcfg.validation_batch,
                                            cmcfg.image_size,
                                            cmcfg.model_image_size[:2],
                                            num_epochs=None,
                                            return_heatmaps=False)
            val_images = inception_input_processing(val_images)

    # Make model
    inception_kwargs = {
        'dropout_keep_prob': incfg.dropout_keep_prob,
    }
    if scope:
        inception_kwargs['scope'] = scope
    if 'InceptionV3/Logits' not in incfg.exclude_scopes:
        # To reaload the logits, we need this, since the pretrained model has
        # 1001 classes (an extra null class)
        inception_kwargs['num_classes'] = 1001
    with slim.arg_scope(inception.inception_v3_arg_scope(
            weight_decay=incfg.weight_decay,
            stddev=incfg.truncated_normal_initializer_stdev)):
        train_logits, train_endpoints = inception.inception_v3(train_images, is_training=training,
                                                               modulators=train_heatmaps if use_modulation else None,
                                                               reuse=reuse,
                                                               **inception_kwargs)
        # Validation model dupes above with reuse=True
        if validation_dir is not False:
            val_logits, val_endpoints = inception.inception_v3(val_images, reuse=True,
                                                               **inception_kwargs)
    if 'InceptionV3/Logits' not in incfg.exclude_scopes:
        # Here, we are keeping the model's 1001 classes, so shift to our data
        train_endpoints['Predictions'] -= 0
        if validation_dir is not False:
            val_endpoints['Predictions'] -= 0
    # Figure out which variables to restore
    vars_to_restore = []
    for var in slim.get_model_variables():
        if not any(map(lambda x: var.op.name.startswith(x), incfg.exclude_scopes)):
            vars_to_restore.append(var)
    restorer = tf.train.Saver(vars_to_restore)
    saver = tf.train.Saver()

    # Loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_logits, labels=train_labels))
    if use_attention_gradients:
        loss = add_attention_gradient_loss(loss, train_logits, train_labels, train_images, train_heatmaps, cmcfg)

    # Training, metrics
    train_op = inception_finetune_learning(loss, cmcfg, incfg)
    train_acc = class_accuracy(train_endpoints['Predictions'], train_labels)
    if validation_dir is not False:
        val_acc = class_accuracy(val_endpoints['Predictions'], val_labels)

    # Summary stuff
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('TrainAcc', train_acc)
    if validation_dir is not False:
        tf.summary.scalar('ValAcc', val_acc)
    summary_op = tf.summary.merge_all()

    # Please update callers before changing this list of returns. Callers so far:
    # `baseline_train_inception.py`, `attention_gradient_train_inception.py`
    if validation_dir is not False:
        return (train_logits, train_endpoints, train_op, summary_op, out_dir,
                loss, restorer, saver, validation_data, val_acc, dt_stamp,
                train_acc)
    else:
        return (train_logits, train_endpoints, train_op, summary_op, out_dir,
                loss, restorer, saver, dt_stamp, train_acc)
