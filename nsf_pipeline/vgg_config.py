import os
import tensorflow as tf

class ConfigVgg():
    def __init__(self, **kwargs):

        # directories for storing tfrecords, checkpoints etc.
        self.main_dir = '/media/data_cifs/andreas/clicktionary_nsf/'
        self.tfrecords = os.path.join(self.main_dir, 'tfrecords')
        self.sampled_indicies = os.path.join(self.main_dir, 'sampled_indicies')
        self.caffe_weights = '/media/data_cifs/clicktionary/pretrained_weights/vgg16.npy'

        self.batch_size = 1
        self.output_shape = 1000 # output shape of the model if 2 we have binary classification 
        self.input_image_size = [256, 256, 3] # size of the input tf record image
        self.resize_to = [256, 256] # option to resize the tf record to a specified size
        self.model_image_size = [224,224, 3] # image dimesions that the model takes in   


        """
        CHECKPOINTS:
        /media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0003_145928_2017_09_11_11_23_49/ 
        /media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0003_145928_2017_09_11_11_23_42/
        /media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0003_50000_2017_09_08_18_16_10/
        /media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0003_50000_2017_09_08_18_16_02/
        /media/data_cifs/clicktionary/clickme_experiment/attgrad_vgg_checkpoints/gradient_0003_50000_2017_09_08_18_15_26/
        """

    
        