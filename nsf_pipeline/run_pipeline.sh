#!/usr/bin/env bash
DEVICE="$1"
CHECKPOINT="$2"
CHECKPOINT_NAME="$(echo $CHECKPOINT | cut -f7 -d'/')"
MAIN_DIR="/media/data_cifs/andreas/clicktionary_nsf/"
TRAIN_NAME="_train_data"
TEST_NAME="_test_data"
RESUTLS_NAME="_results"
TRAIN_DIR=$MAIN_DIR$CHECKPOINT_NAME$TRAIN_NAME
TEST_DIR=$MAIN_DIR$CHECKPOINT_NAME$TEST_NAME
RESULTS_DIR=$CHECKPOINT_NAME$RESUTLS_NAME
python vgg16_feature_extract.py $DEVICE $TRAIN_DIR $TEST_DIR $CHECKPOINT
python train_and_test_SVM.py conv1_1 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv1_2 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv2_1 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
wait
python train_and_test_SVM.py conv2_2 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv3_1 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv3_2 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
wait
python train_and_test_SVM.py conv3_3 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv4_1 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv4_2 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
wait
python train_and_test_SVM.py conv4_3 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv5_1 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py conv5_2 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
wait
python train_and_test_SVM.py conv5_3 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py fc6 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
python train_and_test_SVM.py fc7 $TRAIN_DIR $TEST_DIR $RESULTS_DIR &
wait
python load_human_data_and_compute_correlations.py conv1_1 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv1_2 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv2_1 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv2_2 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv3_1 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv3_2 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv3_3 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv4_1 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv4_2 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv4_3 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv5_1 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv5_2 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py conv5_3 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py fc6 $RESULTS_DIR &
python load_human_data_and_compute_correlations.py fc7 $RESULTS_DIR &
wait
python cleanup.py accuracy $CHECKPOINT_NAME$RESULTS_DIR 
python cleanup.py correlation $CHECKPOINT_NAME$RESULTS_DIR 
Rscript create_layers_plot.R $CHECKPOINT_NAME$RESULTS_DIR
