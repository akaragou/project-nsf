#!/usr/bin/env python
from __future__ import division
import glob
import numpy as np
from tf_record import create_tf_record

def iterate_files(img_and_labels, imgs, file_pointers, labels):
    """
    Reads data from img_and_labels and imgs and loads the file pointers
    and labels into their corresponding arrays
    input: text_file_array, image_array 
    output: None
    """
    labels_dic = {}
    for line in img_and_labels:
        split_line = line.strip('\n').split('\t') # text file is seperated by tabs
        img = split_line[0]
        label = split_line[1]

        labels_dic[img] = int(label)
        
    for f in imgs:
        file_name = f.split('/')[-1]
        image_name = file_name.split('.')[0]

        if image_name in labels_dic:
            file_pointers += [f]
            labels += [labels_dic[image_name]]

def build_train_tfrecords(tfrecords_filename):
    """
    builds tensorflow records (binary files) 
    Input: tfrecords_filename - Filepath where images will be stored
    Ouptut: None
    """

    all_images = glob.glob('/media/data_cifs/nsf_levels/TURK_IMS/trainset_bw/*.jpg') # gathering all images filepaths 
                                                                                     # in an array with the extension .jpg
   
    with open('training_labels.txt') as f:
        label_file_array = f.readlines()

    file_pointers = []
    labels = [] # an array of binary lables to be populated 1 for animal 0 for non animal

    print "Creating training pointers and labels..."
    iterate_files(label_file_array, all_images, file_pointers,labels)
    print  "Done creating training pointers and labels!"
   
    create_tf_record(tfrecords_filename, file_pointers, labels, is_grayscale=True, resize=False)

def build_test_tfrecords(tfrecords_filename):
    """
    builds tensorflow records (binary files) 
    Input: tfrecords_filename - Filepath where images will be stored
    Ouptut: None
    """

    # all_images_0 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_0/*.jpg")
    all_images_30 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_30/*.jpg")
    all_images_31 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_31/*.jpg")
    all_images_32 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_32/*.jpg")
    all_images_33 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_33/*.jpg")
    all_images_34 = glob.glob("/media/data_cifs/nsf_levels/TURK_IMS/set1603241729_34/*.jpg")

    with open('test_labels.txt') as f:
        label_file_array = f.readlines()

    file_pointers = []
    labels = [] # an array of binary lables to be populated 1 for animal 0 for non animal
    
    all_images = all_images_30 + all_images_31 + all_images_32 + all_images_33 + all_images_34 # gathering all images filepaths 
                                                                                               # in an array with the extension .jpg

    print "Creating test pointers and labels..."
    iterate_files(label_file_array, all_images, file_pointers,labels)
    print  "Done creating test pointers and labels!"

    create_tf_record(tfrecords_filename, file_pointers, labels, is_grayscale=True, resize=False)

if __name__ == '__main__':
    # build_train_tfrecords('/media/data_cifs/andreas/clicktionary_nsf/tfrecords/train.tfrecords')
    build_test_tfrecords('/media/data_cifs/andreas/clicktionary_nsf/tfrecords/test.tfrecords')
