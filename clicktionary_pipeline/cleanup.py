import glob
import csv
import argparse
import os

def cleanup_accuracy_results(type, results_dir):
    """
    Creates a csv containing Layer, Accuracy results or Layer, Correlation results
    Input: type - a type indicating which csv we are creating either one for all the 
           accuracy results or one for all the correlation results
           results_dir - a directory to where the layer results are located
    Ouput: None 
    """
    if type == "accuracy":
        header = ['Layers', 'Accuracy']
        layers = ['conv1_1', 'conv1_2',\
                  'conv2_1', 'conv2_2',\
                  'conv3_1', 'conv3_2', 'conv3_3',\
                  'conv4_1', 'conv4_2', 'conv4_3',\
                  'conv5_1', 'conv5_2', 'conv5_3',\
                  'fc6','fc7']

        with open(os.path.join(results_dir,'layer_accuracy_results.csv'), 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for layer in layers:
                for file_name in [f for f in glob.glob(results_dir+'/*.txt') if "accuracy" in f]: # iterating through accuracy results txt files
                    full_name = file_name.split('/')[-1] 
                    conv_names = full_name.split('_')[0] + '_' + full_name.split('_')[1]
                    fully_connect_names = full_name.split('_')[0]
                    if layer == conv_names:
                        with open(file_name, 'r') as f:
                            first_line = f.readline()
                            accuracy = float(first_line.strip(' ').split(':')[1])
                            writer.writerow([layer, accuracy])
                    elif layer == fully_connect_names:
                        with open(file_name, 'r') as f:
                            first_line = f.readline()
                            accuracy = float(first_line.strip(' ').split(':')[1])
                            writer.writerow([layer, accuracy])

    if type == "correlation":
        header = ['Layers', 'Correlation']
        layers = ['conv1_1', 'conv1_2',\
                  'conv2_1', 'conv2_2',\
                  'conv3_1', 'conv3_2', 'conv3_3',\
                  'conv4_1', 'conv4_2', 'conv4_3',\
                  'conv5_1', 'conv5_2', 'conv5_3',\
                  'fc6','fc7']

        with open(os.path.join(results_dir,'layer_correlation_results.csv'), 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for layer in layers:
                for file_name in [f for f in glob.glob(results_dir+'/*.txt') if "correlation" in f]: # iterating through correlation results txt files
                    full_name = file_name.split('/')[-1] 
                    conv_names = full_name.split('_')[0] + '_' + full_name.split('_')[1]
                    fully_connect_names = full_name.split('_')[0]
                    if layer == conv_names:
                        with open(file_name, 'r') as f:
                            first_line = f.readline()
                            accuracy = float(first_line.strip(' ').split(':')[1])
                            writer.writerow([layer, accuracy])
                    elif layer == fully_connect_names:
                        with open(file_name, 'r') as f:
                            first_line = f.readline()
                            accuracy = float(first_line.strip(' ').split(':')[1])
                            writer.writerow([layer, accuracy])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('type')
    parser.add_argument('results_dir')
    args = parser.parse_args()

    cleanup_accuracy_results(args.type, args.results_dir)