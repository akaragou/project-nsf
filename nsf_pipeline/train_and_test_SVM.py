from __future__ import division
from sklearn.svm import LinearSVC
import numpy as np
import os
import argparse

def make_dir(dir):
    """
    creates a directory if it does not exist
    Input: a directory
    Output: None
    """
    if not os.path.isdir(dir): os.makedirs(dir)

def load_train_data(train_dir, batch, layer):
    """
    Loads train features and labels for a specific batch
    Input: train_dir - directory where train batch data is located
           batch - batch name to load, example: 'batch_1_' ...
           layer - layer to load batch data for, example: conv1_1 ...
    Ouput: X_train_batch.tolist() - a list of train features for a specific layer and batch
           y_train_batch.tolist() - a list of train labels for a specific layer and batch
    """
    batch_layer = batch + layer + '.npy'
    batch_layer_filename = os.path.join(train_dir,batch_layer)
    X_train_batch = np.load(batch_layer_filename)

    batch_label = batch + 'train_labels.npy'
    batch_label_filename = os.path.join(train_dir,batch_label)
    y_train_batch = np.load(batch_label_filename)

    return X_train_batch.tolist(), y_train_batch.tolist()


def load_test_data(test_dir, layer):
    """
    Loads test features and labels
    Input: test_dir - a directory containing test features
           layer - a layer to optimize for, example: conv1_1 ...
    Ouptut: X_test - a numpy array of test features
            y_test - a numpy array of test labels
            test_file_path - the full filepath to the test images
    """
    layer_filename = layer + '.npy'
    X_test = np.load(os.path.join(test_dir, layer_filename))
    y_test = np.load(os.path.join(test_dir,'test_labels.npy'))
    test_file_path = np.load(os.path.join(test_dir,'test_file_paths.npy'))
    return X_test, y_test, test_file_path


def normalize_train_data(matrix):
    """
    Performs Z-score normalization on the train data 
    Input: matrix - a numpy array of dimensions nxm where n are the number 
                    of observations and m are the number of features
    Output: normalized_matrix.T - the normalize matrix with original input dimensions
            params - the train data mean and standard deviation stored in a tuple to be 
                    used when normalizing the test data 
    """
    normalized_matrix = []
    params = []
    count = 0
    for feat in matrix.T:
        std = feat.std()
        if std == 0: # some features do not vary, avoiding division by zero
            std = 1
            count +=1
        mean = feat.mean()
        obs_per_feat = []
        for obs in feat:
            obs_per_feat.append((obs-mean)/std)
        normalized_matrix.append(obs_per_feat)
        params.append((mean, std))
    normalized_matrix = np.array(normalized_matrix)
    print "numbers of features with std 0: {0}".format(count)
    return normalized_matrix.T, params

def normalize_test_data(matrix, params):
    """
    Performing Z-score normalization on the test data with the train data mean and standard deviation
    Input: matrix - a numpy array of dimensions nxm where n are the number of obserservations and m
                    are the number of features
           params - a tuple of train data mean and standard deviation
    Output: normalized_matrix.T - the normalize matrix with original input dimensions
    """
    normalized_matrix = []
    for feat, param in zip(matrix.T, params):
        mean, std = param
        obs_per_feat = []
        for obs in feat:
            obs_per_feat.append((obs-mean)/std)
        normalized_matrix.append(obs_per_feat)
    normalized_matrix = np.array(normalized_matrix)
    return normalized_matrix.T

def train_and_test_SVM(results_numpy, train_dir, test_dir, layer):
    """
    Training and testing SVM and storing numpy array with results
    Input: results_numpy - numpy array file name to store results to
           train_dir - directory where SVM train data is located
           test_dir - directory where SVM test data is located
           layer - layer to train SVM on, example: conv1_1 ...
    Output: accuracy_msg - a print statement containing the accuracy results of the test data
    """   

    batches = ['batch_1_','batch_2_','batch_3_', 'batch_4_'] 
    X_train = []
    y_train = []
    for batch in batches:
        print "Loading Train: {0}...".format(batch[:-1])
        X_train_batch, y_train_batch = load_train_data(train_dir, batch, layer)
        X_train = X_train + X_train_batch
        y_train = y_train + y_train_batch
        print "Done loading Train: {0}!".format(batch[:-1])

    X_train = np.array(X_train)
    y_train = np.array(y_train)


    print "Loading Test data.."
    X_test, y_test, test_file_names = load_test_data(test_dir, layer)
    print "Done loading Test data!"

    print "Normalizing train data..."
    X_train, params = normalize_train_data(X_train) 
    print "Done normalizing train data!"
    print "Normalizing test sets..."
    X_test = normalize_test_data(X_test, params)
    print "Done Normalizing test sets!"


    clf = LinearSVC(dual = False, C = 1e-05)
    print "Training SVM..."
    clf.fit(X_train,y_train)
    print "Done Training SVM..."

    predictions = clf.predict(X_test)
    dist = clf.decision_function(X_test)
    np.save(results_numpy, [test_file_names, dist, predictions, y_test]) # storing the test image full filepaths
                                                                         # distance to the hyperplane
                                                                         # predicted label and true label
    true_count = 0
    for true, pred in zip(y_test, predictions):
        if true == pred:
            true_count += 1

    accuracy = true_count/len(y_test)
    accuracy_msg = "Test accuracy for sets 30 - 34: {0}".format(accuracy)
    total_correct_msg = "Total correct: {0}/{1}".format(true_count,len(y_test))
    print accuracy_msg
    print total_correct_msg
    print

    return accuracy_msg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('layer')
    parser.add_argument('train_dir')
    parser.add_argument('test_dir')
    parser.add_argument('results_dir')
    args = parser.parse_args()

    make_dir(args.results_dir)

    results_numpy = args.results_dir + args.layer + '_results.npy'
    accuracy = train_and_test_SVM(results_numpy, args.train_dir, args.test_dir, args.layer)

    results_txt = args.results_dir + args.layer + '_accuracy_results.txt'
    with open(results_txt, 'wt') as f:
        f.write(accuracy +'\n')


   
    