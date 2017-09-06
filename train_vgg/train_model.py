import os
import re
import sys
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from data_loader import inputs
from ops.tf_fun import make_dir, training_loop
from ops.loss_utils import softmax_loss , wd_loss
from ops.metrics import class_accuracy
from config import vggConfig
import vgg_model as vgg16
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_vgg16():

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
    config = vggConfig()
    train_data = config.training_images
    train_meta = np.load(config.training_meta)
    print 'Using train tfrecords: %s | %s image/heatmap combos' % (
            [train_data], len(train_meta['labels']))

    validation_data = config.validation_images
    val_meta = np.load(config.validation_meta)
    print 'Using validation tfrecords: %s | %s images' % (
        validation_data, len(val_meta['labels']))
    # Make output directories if they do not exist
    dt_stamp = 'baseline_' +\
        str(config.initial_learning_rate)[2:] + '_' + str(
            len(train_meta['labels'])) + '_' + re.split(
            '\.', str(datetime.now()))[0].\
        replace(' ', '_').replace(':', '_').replace('-', '_')
    config.train_checkpoint = os.path.join(
        config.train_checkpoint, dt_stamp)  # timestamp this run
    out_dir = os.path.join(config.results, dt_stamp)
    dir_list = [
        config.train_checkpoint, config.train_summaries,
        config.results, out_dir]
    [make_dir(d) for d in dir_list]

    print '-'*60
    print('Training model:' + dt_stamp)
    print '-'*60

    # Prepare data on CPU
    with tf.device('/cpu:0'):
        train_images, train_labels = inputs(
            train_data, config.train_batch, config.image_size,
            config.model_image_size[:2],
            train=config.data_augmentations,
            num_epochs=config.epochs,
            return_heatmaps=False)
        val_images, val_labels = inputs(
            validation_data, config.validation_batch, config.image_size,
            config.model_image_size[:2],
            num_epochs=None,
            return_heatmaps=False)

        step = get_or_create_global_step()
        step_op = tf.assign(step, step+1)
    # Prepare model on GPU
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn') as scope:
            vgg = vgg16.model_struct()
            train_mode = tf.get_variable(name='training', initializer=True)
            vgg.build(
                train_images,
                is_training=True,
                batchnorm=True)

            # Prepare the loss function
            loss = softmax_loss(logits=vgg.fc8, labels=train_labels)

            # Add weight decay of fc6/7/8
            if config.wd_penalty is not None:
                loss = wd_loss(
                    loss=loss,
                    trainables=tf.trainable_variables(),
                    config=config)

            lr = tf.train.exponential_decay(
                        learning_rate = config.initial_learning_rate,
                        global_step = step_op,
                        decay_steps = config.decay_steps,
                        decay_rate = config.learning_rate_decay_factor,
                        staircase = True)

            if config.optimizer == "adam":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.AdamOptimizer(lr).minimize(loss)
            elif config.optimizer == "sgd":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.GradientDescentOptimizer(lr).minimize(loss)
            elif config.optimizer == "nestrov":
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op =  tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss)
            else:
                raise Exception("Not known optimizer! options are adam, sgd or nestrov")

            train_accuracy = class_accuracy(vgg.prob, train_labels)  # training accuracy

            # Add summaries for debugging
            tf.summary.image('train images', train_images)
            tf.summary.image('validation images', val_images)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("training accuracy", train_accuracy)

            # Setup validation op
            scope.reuse_variables()

            # Validation graph is the same as training except no batchnorm
            val_vgg = vgg16.model_struct()
            val_vgg.build(val_images,
                        is_training=False,
                        batchnorm=True)

            # Calculate validation accuracy
            val_accuracy = class_accuracy(val_vgg.prob, val_labels)
            tf.summary.scalar("validation accuracy", val_accuracy)

    # Set up summaries and saver
    saver = tf.train.Saver(
        tf.global_variables(), max_to_keep=config.keep_checkpoints)
    print
    print "Variables stored in checpoint:"
    print_tensors_in_checkpoint_file(file_name='/media/storage/andreas/vgg16_grayscale_train/checkpoints/baseline_0001_1283163_2017_08_17_21_00_31/model_425000.ckpt-425000', tensor_name='',all_tensors='')
    conv_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/vgg_conv/')
    fc6_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/fc6/')
    fc6_batchnorm = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/fc6_batchnorm')
    fc7_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/fc7/')
    fc7_batchnorm = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/fc7_batchnorm')
    fc8_vriables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cnn/fc8/')
    model_variables = conv_variables + fc6_variables + fc6_batchnorm + fc7_variables + fc7_batchnorm + fc8_vriables
    print "Model variables to restore:"
    for var in model_variables:
        print var
    print
    restorer = tf.train.Saver(model_variables)
    summary_op = tf.summary.merge_all()

    # Initialize the graph
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # Need to initialize both of these if supplying num_epochs to inputs
    sess.run(tf.group(tf.global_variables_initializer(),
             tf.local_variables_initializer()))
    restorer.restore(sess, '/media/storage/andreas/vgg16_grayscale_train/checkpoints/baseline_0001_1283163_2017_08_17_21_00_31/model_425000.ckpt-425000')
    summary_dir = os.path.join(
        config.train_summaries, dt_stamp)
    summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

    # Set up exemplar threading
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # Start training loop
    np.save(os.path.join(out_dir, 'training_config_file'), config)
    training_loop(
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
        lr)


if __name__ == '__main__':
    train_vgg16()
