import os

class vggConfig():
    def __init__(self, **kwargs):
        # Image directories
        self.image_base_path = '/media/data_cifs/clicktionary/clickme_experiment/tf_records/'        
        self.training_images = os.path.join(
            self.image_base_path, 'full_imagenet_with_some_clicks_train_7.tfrecords')
        self.validation_images = os.path.join(
            self.image_base_path, 'imagenet_val.tfrecords')
        self.training_meta = os.path.join(self.image_base_path, 
            'full_imagenet_with_some_clicks_train_7_meta.npz')
        self.validation_meta = os.path.join(self.image_base_path,
             'imagenet_val_meta.npz')
        self.im_ext = '.JPEG'

        self.project_base_path = '/media/data_cifs/andreas/vgg_train/'
        self.results = os.path.join(self.project_base_path, 'results')
        self.train_checkpoint = os.path.join(
            self.project_base_path, 'checkpoints')
        self.train_summaries = os.path.join(
            self.project_base_path, 'summaries')
    
        self.image_size = [256, 256, 3]
        self.train_batch = 32
        self.validation_batch = 32
        self.optimizer = "adam"
        self.initial_learning_rate = 1e-05
        self.momentum = 0.95 # if optimizer is nestrov
        self.decay_steps = 100000
        self.learning_rate_decay_factor = 0.5
        # validation_batch * num_validation_evals is num of val images to test
        self.num_validation_evals = 100
        self.validation_iters = 1000  # test validation every this # of steps
        self.epochs = 1000 # 400  # Increase since we are augmenting
        self.top_n_validation = 0  # set to 0 to save all
        self.model_image_size = [224, 224, 3]
        self.output_shape = 1000  # how many categories for classification
        self.wd_layers = ['conv1_1','conv1_2',
                        'conv2_1','conv2_2',
                        'conv3_1','conv3_2','conv3_3',
                        'conv4_1','conv4_2','conv4_3',
                        'conv5_1','conv5_2','conv5_3',
                        'fc6', 'fc7', 'fc8']
        self.keep_checkpoints = 100  # max # of checkpoints
        self.wd_penalty = 5e-5

        self.data_augmentations = [
            'random_crop', 'left_right',
        ]

        # update attributes
        self.__dict__.update(kwargs)

