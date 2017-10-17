from __future__ import division
import numpy as np
import tensorflow as tf
import gc

VGG_MEAN = [103.939, 116.779, 123.68] # color means 

class model_struct:
    """
    A trainable version VGG16.
    """

    def __init__(self,
                weight_npy_path=None, trainable=True,
                fine_tune_layers=None):
        if weight_npy_path is not None:
            self.data_dict = np.load(weight_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.joint_label_output_keys = []

    def __getitem__(self, name):
        return getattr(self, name)

    def __contains__(self, name):
        return hasattr(self, name)

    def build(
            self,
            img,
            is_training=False,
            is_grayscale=False,
            batchnorm=False,
            ):
        """
        load variable from npy to build the VGG

        :param grayscale: grayscale image [batch, height, width, 1] values scaled [0, 1]
        :param train_mode: a bool tensor, usually a placeholder:
        :if True, dropout will be turned on
        """

        if is_grayscale:
            grayscale_img = tf.identity(img, name="input_grayscale_img")
            grayscale_img_scaled = grayscale_img * 255.0   
            normalized_img = grayscale_img_scaled - 117.257 #grayscale mean 
        else:
            color_img = tf.identity(img, name="input_color_img")
            image_rgb_scaled = color_img * 255.0
            red, green, blue = tf.split(num_or_size_splits=3, axis=3, value=image_rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            normalized_img = tf.concat(values = [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
                ], axis=3)
            assert normalized_img.get_shape().as_list()[1:] == [224, 224, 3], normalized_img.get_shape().as_list()
            

        layer_structure = [
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [64, 64, None],
                'names': ['conv1_1', 'conv1_2', 'pool1'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'pool'],
                'weights': [128, 128, None],
                'names': ['conv2_1', 'conv2_2', 'pool2'],
                'filter_size': [3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'conv', 'pool'],
                'weights': [256, 256, 256, None],
                'names': ['conv3_1', 'conv3_2', 'conv3_3', 'pool3'],
                'filter_size': [3, 3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'conv', 'pool'],
                'weights': [512, 512, 512, None],
                'names': ['conv4_1', 'conv4_2', 'conv4_3', 'pool4'],
                'filter_size': [3, 3, 3, None]
            },
            {
                'layers': ['conv', 'conv', 'conv', 'pool'],
                'weights': [512, 512, 512, None],
                'names': ['conv5_1', 'conv5_2', 'conv5_3', 'pool5'],
                'filter_size': [3, 3, 3, None]
            }
            ]

        output_conv = self.create_conv_tower(
            normalized_img,
            layer_structure,batchnorm,
            is_training, tower_name='vgg_conv')
       
        with tf.variable_scope('vgg_fc'):
            output_shape = [int(x) for x in output_conv.get_shape()]
            flattened_conv = tf.reshape(
                output_conv,
                [output_shape[0], np.prod(output_shape[1:])])
            # FC6
            self.fc6 = self.fc_layer(
                bottom=flattened_conv,
                in_size=int(flattened_conv.get_shape()[-1]),
                out_size=4096,
                name='fc6')
           
            if batchnorm:
                self.fc6 = tf.contrib.layers.batch_norm(self.fc6, is_training=is_training, scope='fc6_batchnorm')
            self.fc6 = tf.layers.dropout(self.fc6, rate=0.5, training=is_training)
            print 'Added layer: fc6'

            # FC7
            self.fc7 = self.fc_layer(
                bottom=self.fc6,
                in_size=int(self.fc6.get_shape()[-1]),
                out_size=4096,
                name='fc7')

            if batchnorm:
                self.fc7 = tf.contrib.layers.batch_norm(self.fc7, is_training=is_training, scope='fc7_batchnorm')
            self.fc7 = tf.layers.dropout(self.fc7, rate=0.5, training=is_training)
            print 'Added layer: fc7'
            # FC8
            self.fc8 = self.fc_layer(
                bottom=self.fc7,
                in_size=int(self.fc7.get_shape()[-1]),
                out_size=1000,
                name='fc8')
            print 'Added layer: fc8'

        self.final = tf.identity(self.fc8, name="output")
        self.prob = tf.nn.softmax(self.fc8)
        self.data_dict = None

    def create_conv_tower(self, act, layer_structure,batchnorm,is_training,tower_name):
        print 'Creating tower: %s' % tower_name
        with tf.variable_scope(tower_name):
            for layer in layer_structure:
                for la, we, na, fs in zip(
                        layer['layers'],
                        layer['weights'],
                        layer['names'],
                        layer['filter_size']):
                    if la == 'pool':
                        act = self.max_pool(
                            bottom=act,
                            name=na)
                    elif la == 'conv':
                        act = self.conv_layer(
                            bottom=act,
                            in_channels=int(act.get_shape()[-1]),
                            out_channels=we,
                            name=na,
                            batchnorm=batchnorm,
                            is_training=is_training,
                            filter_size=fs
                        )
                    setattr(self, na, act)
                    print 'Added layer: %s' % na
        return act

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(
            bottom, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(
                    self, bottom, in_channels,
                    out_channels, name, batchnorm,is_training, 
                    filter_size=3, stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(
                filter_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            if batchnorm:
                relu = tf.contrib.layers.batch_norm(relu, is_training=is_training, 
                                                    scope =name + '_batchnorm')

            return relu

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(
            self, filter_size, in_channels, out_channels,
            name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [filter_size, filter_size, in_channels, out_channels],
                tf.contrib.layers.xavier_initializer_conv2d(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [filter_size, filter_size, in_channels, out_channels],
                0.0, 0.001)
        bias_init = tf.truncated_normal([out_channels], .0, .001)
        filters = self.get_var(weight_init, name, 0, name + "_filters")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name, init_type='xavier'):
        if init_type == 'xavier':
            weight_init = [
                [in_size, out_size],
                tf.contrib.layers.xavier_initializer(uniform=False)]
        else:
            weight_init = tf.truncated_normal(
                [in_size, out_size], 0.0, 0.001)
        bias_init = tf.truncated_normal([out_size], .0, .001)
        weights = self.get_var(weight_init, name, 0, name + "_weights")
        biases = self.get_var(bias_init, name, 1, name + "_biases")

        return weights, biases

    def get_var(
            self, initial_value, name, idx,
            var_name, in_size=None, out_size=None):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            # get_variable, change the boolian to numpy
            if type(value) is list:
                var = tf.get_variable(
                    name=var_name, shape=value[0], initializer=value[1])
            else:
                var = tf.get_variable(name=var_name, initializer=value)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        return var

    def save_npy(self, sess, npy_path="./vgg16-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}
        num_files = 0

        for (name, idx), var in self.var_dict.items():
            # print(var.get_shape())
            if name == 'fc6':
                np.save(npy_path+str(num_files), data_dict)
                data_dict.clear()
                gc.collect()
                num_files += 1
                for i, item in enumerate(tf.split(var, 8, 0)):
                    # print(i)
                    name = 'fc6-'+ str(i)
                    if name not in data_dict.keys():
                        data_dict[name] = {}
                    data_dict[name][idx] = sess.run(item)
                    np.save(npy_path+str(num_files), data_dict)
                    data_dict.clear()
                    gc.collect()
                    num_files += 1
            else :
                var_out = sess.run(var)
                if name not in data_dict.keys():
                    data_dict[name] = {}
                data_dict[name][idx] = var_out

        np.save(npy_path+str(num_files), data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count