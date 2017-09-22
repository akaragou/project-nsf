#!/usr/bin/env python
from __future__ import division
import tensorflow as tf
import os
import argparse
import numpy as np
import random
import matplotlib.pyplot as plot_layers
from vgg_config import ConfigVgg
from baseline_vgg16 import Vgg16
from tf_record import tfrecord2metafilename, read_and_decode
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def make_dir(dir):
    """
    creates a directory if it does not exist
    Input: a directory
    Output: None
    """
    if not os.path.isdir(dir): os.makedirs(dir)

def create_conv_features(layer, sampled_indicies):
    """
    creates a sample convolutional layer based on the 4096 indicies to sample from 
    Input: layer - a convolutional layer 
           sampled_indicies - an array of 4096 indicies to sample from 
    Output: a sampled convolutional layer of len 4096
    """
    flattened_layer = layer.ravel()
    sample = [flattened_layer[i] for i in sampled_indicies]
    return sample


def update_file_names(main_directory, batch):
    """
    creates array of file paths for each convolutional layer and fully connected layer to
    be later used for storing the features 
    Input: main_directory - main directory to store convolutional and fully connected layers in 
           batch - the current batch number
    Output: an array of filepaths 
    """
    conv1_1_sample_file = main_directory + '/' + batch + 'conv1_1.npy'
    conv1_2_sample_file = main_directory + '/' + batch + 'conv1_2.npy'

    conv2_1_sample_file = main_directory + '/' + batch + 'conv2_1.npy'
    conv2_2_sample_file = main_directory + '/' + batch + 'conv2_2.npy'

    conv3_1_sample_file = main_directory + '/' + batch + 'conv3_1.npy'
    conv3_2_sample_file = main_directory + '/' + batch + 'conv3_2.npy'
    conv3_3_sample_file = main_directory + '/' + batch + 'conv3_3.npy'

    conv4_1_sample_file = main_directory + '/' + batch + 'conv4_1.npy'
    conv4_2_sample_file = main_directory + '/' + batch + 'conv4_2.npy'
    conv4_3_sample_file = main_directory + '/' + batch + 'conv4_3.npy'

    conv5_1_sample_file = main_directory + '/' + batch + 'conv5_1.npy'
    conv5_2_sample_file = main_directory + '/' + batch + 'conv5_2.npy'
    conv5_3_sample_file = main_directory + '/' + batch + 'conv5_3.npy'

    fc6_file =  main_directory + '/' +  batch + 'fc6.npy'
    fc7_file = main_directory + '/' + batch + 'fc7.npy'

    file_array = [conv1_1_sample_file, conv1_2_sample_file,\
                  conv2_1_sample_file, conv2_2_sample_file,\
                  conv3_1_sample_file, conv3_2_sample_file, conv3_3_sample_file,\
                  conv4_1_sample_file, conv4_2_sample_file, conv4_3_sample_file,\
                  conv5_1_sample_file, conv5_2_sample_file, conv5_3_sample_file,\
                  fc6_file, fc7_file]

    return file_array

def vgg16_train_feature_extract(device, config, model_path=None):
    """
    extracting train features for each convolutional and fully connected layer and storing them as numpy arrays in batches 
    Input: device - a gpu device
           config - a config file containing values for filepaths and model parameters
    Output: None
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    with tf.Graph().as_default():
        
        # loading training data
        train_fn = os.path.join(config.tfrecords,'train.tfrecords')
        train_meta = np.load(tfrecord2metafilename(train_fn))
        print ('Using train tfrecords: %s | %s images' % (train_fn, len(train_meta['labels'])))
        
        filename_queue = tf.train.string_input_producer([train_fn], num_epochs=1)

        train_image, train_label, file_path = read_and_decode(filename_queue=filename_queue, img_dims=config.input_image_size, 
                                                              resize_to=config.resize_to, model_dims=config.model_image_size, 
                                                              size_of_batch=config.batch_size,labels=True, augmentations_dic=None, 
                                                              num_of_threads=1, shuffle=False)
   
        # building vgg model 
        with tf.variable_scope('cnn'):
            vgg = Vgg16(config.caffe_weights)
            validation_mode = tf.Variable(False, name='training')
            vgg.build(train_image, output_shape=config.output_shape,train_mode=validation_mode) 


        layers_and_vals = [vgg.conv1_1, vgg.conv1_2,\
                            vgg.conv2_1, vgg.conv2_2,\
                            vgg.conv3_1, vgg.conv3_2, vgg.conv3_3,\
                            vgg.conv4_1, vgg.conv4_2, vgg.conv4_3,\
                            vgg.conv5_1,  vgg.conv5_2,  vgg.conv5_3,\
                            vgg.fc6, vgg.fc7, vgg.prob, file_path, train_label]

        print "Variables stored in checpoint:"
        print_tensors_in_checkpoint_file(file_name=model_path, tensor_name='',all_tensors='')
        restorer = tf.train.Saver(tf.global_variables())
        # Initialize the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Need to initialize both of these if supplying num_epochs to inputs
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            if model_path is not None:
                restorer.restore(sess, model_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            count = 0 # count for each image
            batch_index = 0 # index for each batch
            
            range_of_indicies = [] # an array for the range of possible indicies in each convolutional layer 
            conv_layers = layers_and_vals[:13] # convolutional layers in vgg16 are the first 13 layers
            feature_array = [[] for _ in range(15)]

            for layer in conv_layers: 
                flatten_dim_len =  reduce(lambda x, y: x*y, layer.get_shape()[1:]) # getting the flatten dimension of each convolutional layer
                range_of_indicies.append(range(flatten_dim_len)) 

            sampled_indicies = []
            # creating the the inidicies that will be sampled for each convolutional layer
            # samples need to be consistent between train features and test features
            for possible_indicies in range_of_indicies:
                indicies = random.sample(possible_indicies, 4096)
                sampled_indicies.append(indicies)

            batch_names = ['batch_1_', 'batch_2_', 'batch_3_', 'batch_4_']

            batches = [20000, 40000, 60000, float('inf')]

            labels = []

            try:

                while not coord.should_stop():

                    np_layers_and_vals = sess.run(layers_and_vals)

                    label = np_layers_and_vals[-1] 
                    file_path = np_layers_and_vals[-2][0]
                    layers = np_layers_and_vals[:15]

                    label = int(label[0])


                    for i in range(len(feature_array)):
                        if i >= 13:
                            feature_array[i].append(np.asarray(np.squeeze(layers[i]),dtype=np.float32)) # appending fully connected layers
                        else:   
                            feature_array[i].append(np.asarray(create_conv_features(layers[i], sampled_indicies[i]),dtype=np.float32)) # appending convolutional layers

                    labels.append(label)

                    msg = "extracting features for image number: {0} || filename: {1} || True label is: {2}"
                    print(msg.format((count + 1), file_path, experiment_labels[label]))  
                    count += 1



                    if count == batches[batch_index]:
                        
                        print 
                        print "Saving batch number {0}...".format(batch_index + 1)
                        file_array = update_file_names(config.SVM_train_data,batch_names[batch_index]) # updating filenames
                        for features, f in zip(feature_array, file_array):
                            np.save(f,features)
                        np.save(os.path.join(config.SVM_train_data, (batch_names[batch_index] + 'train_labels.npy')), labels)
                        print "Done saving batch number {0}!".format(batch_index + 1)
                        print 
                        feature_array = [[] for _ in range(15)]
                        labels = []
                        batch_index += 1


            except tf.errors.OutOfRangeError:
                print 
                print "Saving batch number {0}...".format(batch_index + 1)
                # storing last batch of convolutional and fully connected features
                file_array = update_file_names(config.SVM_train_data,batch_names[batch_index])
                for features, f in zip(feature_array, file_array):
                    np.save(f,features)

                np.save(os.path.join(config.SVM_train_data, (batch_names[batch_index] + 'train_labels.npy')), labels)
                print "Done saving batch number {0}!".format(batch_index + 1)
                print 

                print('Done Creating SVM Train Features :)')
            finally:
                coord.request_stop()  
            coord.join(threads)

    return sampled_indicies

def vgg16_test_feature_extract(device, sampled_indicies, config,model_path=None):
    """
    extracting val features for each convolutional and fully connected layer and storing them as numpy arrays in batches 
    Input: device - a gpu device
           sampled_indicies - sampled indicies for each convolutional layer the indicies sampled have to be consistent 
           between the train and test set 
           config - a config file containing values for filepaths and model parameters
    Output: None
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    with tf.Graph().as_default():

        # loading test data
        test_fn = os.path.join(config.tfrecords,'test.tfrecords')
        test_meta = np.load(tfrecord2metafilename(test_fn))
        print ('Using test tfrecords: %s | %s images' % (test_fn, len(test_meta['labels'])))

        filename_queue = tf.train.string_input_producer(
        [test_fn], num_epochs=1)


        test_image, test_label, file_path = read_and_decode(filename_queue=filename_queue, img_dims=config.input_image_size, 
                                                              resize_to=config.resize_to, model_dims=config.model_image_size, 
                                                              size_of_batch=config.batch_size,labels=True, augmentations_dic=None, 
                                                              num_of_threads=1, shuffle=False)

        with tf.variable_scope('cnn'):
            vgg = Vgg16(config.caffe_weights)
            validation_mode = tf.Variable(False, name='training')
            vgg.build(test_image, output_shape=config.output_shape,train_mode=validation_mode)

        layers_and_vals = [vgg.conv1_1, vgg.conv1_2,\
                            vgg.conv2_1, vgg.conv2_2,\
                            vgg.conv3_1, vgg.conv3_2, vgg.conv3_3,\
                            vgg.conv4_1, vgg.conv4_2, vgg.conv4_3,\
                            vgg.conv5_1,  vgg.conv5_2,  vgg.conv5_3,\
                            vgg.fc6, vgg.fc7, vgg.prob, file_path, test_label]
        
        print "Variables stored in checpoint:"
        print_tensors_in_checkpoint_file(file_name=model_path, tensor_name='',all_tensors='')
        restorer = tf.train.Saver(tf.global_variables())
        # Initialize the graph
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # Need to initialize both of these if supplying num_epochs to inputs
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            if model_path is not None:
                restorer.restore(sess, model_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            count = 0

            feature_array = [[] for _ in range(15)]
            labels = []
            filepaths = []

            try:

                while not coord.should_stop():

                    np_layers_and_vals = sess.run(layers_and_vals)

                    label = np_layers_and_vals[-1] 
                    file_path = np_layers_and_vals[-2][0]
                    layers = np_layers_and_vals[:15]

                    label = int(label[0])

                    for i in range(len(feature_array)):
                        if i >= 13:
                            feature_array[i].append(np.asarray(np.squeeze(layers[i]),dtype=np.float32)) # appending fully connected layers
                        else:   
                            feature_array[i].append(np.asarray(create_conv_features(layers[i], sampled_indicies[i]),dtype=np.float32)) # appending convolutional layers
                       

                    msg = "extracting features for image number: {0} || filename: {1} || True label is: {2}"
                    print(msg.format((count + 1), file_path, experiment_labels[label]))
                    labels.append(label)
                    filepaths.append(file_path)
                    count += 1


            except tf.errors.OutOfRangeError:

                print 
                print "Saving files ..."
                file_array = update_file_names(config.SVM_test_data, '') # updating filenames
                for features, f in zip(feature_array, file_array):
                    np.save(f, features)
                np.save(os.path.join(config.SVM_test_data,'test_labels.npy'), labels) 
                np.save(os.path.join(config.SVM_test_data,'test_file_paths.npy'), filepaths)
                print "Done saving files!"
                print 
                print('Done Creating SVM Test Features :)')
            finally:
                coord.request_stop()  
            coord.join(threads)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("device") # select a device
    parser.add_argument("train_dir") # training directory 
    parser.add_argument("test_dir") # test directory 
    parser.add_argument("checkpoint") # checkpoint to load model from
    args = parser.parse_args()
    config = ConfigVgg()

    config.SVM_train_data = args.train_dir
    config.SVM_test_data = args.test_dir

    make_dir(config.SVM_train_data) # create train directory if they do not exist
    make_dir(config.SVM_test_data) # create test directory if they do not exist

    experiment_labels = {1:'animal', 0:'non-animal'}

    sampled_indicies = vgg16_train_feature_extract(args.device,config, args.checkpoint)
    np.save(os.path.join(config.sampled_indicies,'sampled_indicies.npy'),sampled_indicies)
    vgg16_test_feature_extract(args.device,sampled_indicies,config,args.checkpoint)
