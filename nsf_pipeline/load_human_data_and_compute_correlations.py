from __future__ import division
import numpy as np
import sqlite3
from scipy import stats
import pprint
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import json
import re
import argparse
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score


def append_scores(filename,scores_by_participant,scores_by_img,scores):
    """
    Loading human data from a database and populating scores_by_participant and scores_by_img
    dictionary and scores array
    Input: filename - database filename
           scores_by_participant - a dictionary of scores per participant
           scores_by_img - a dictionary of scores per img
           scores - total score across trials
    Output: None
    """
    r = sqlite3.connect(filename).cursor().execute(
                "SELECT datastring FROM placecat WHERE status in (3,4) AND NOT datastring==''").fetchall()
    print "%d participants found in file %s." % (len(r), filename)
    for row in r:
        f = row[0]
        data = json.loads(f)
        # Get experimental trials.
        alltrials = data['data']
        # Ignore training trials
        exptrials = alltrials[10:]
        # Ignore pause screens
        trials = [t['trialdata'] for t in exptrials if t['trialdata']['trial_type'] == 's2stim']

        for t in trials:


            stim = json.loads(t["stimulus"])

        
            if 'examples' in stim["stimulus"]: # Example video not used in evaluation
                continue

            # recover max_rt and im num in block
            max_rt = stim["duration"] - stim["onset"] - 50 # maybe exclude trails that surpased max response time

            img = '/'.join(stim["stimulus"].split('/')[3:-1])
           
            if t['rt'] > 0:
                
                participant = data['workerId']
                if t['rt'] <= 500:
                   
                    score = 0 
                    if t['response'] == t['true_response']:
                        score = 1  
                    else:
                        score = 0
                    scores.append(score)

                    if img not in scores_by_img:
                        scores_by_img[str(img)] = [score]
                    else:
                        scores_by_img[str(img)].append(score)

                    if participant not in scores_by_participant:
                        scores_by_participant[str(participant)] = [score]
                    else: 
                        scores_by_participant[str(participant)].append(score)  
                


def spearmans_rho_per_layer(scores_by_img, layer_file):
    """
    Computes The Spearman's rho correlation between Distance from the Hyperplane and Human Accuracy
    plots the resulting correlation and stores it in a text file
    Input: scores_by_img - a dictionary of scores per image
           layer_file - the layer filename to evalute the correlation for
    Output: None
    """
    results = np.load(layer_file)
    imgs = [res.split('/')[-1].split('.')[0] for res in results[0]]
    
    decisions = results[1]
    true_labels = results[3]
    predicted_labels = results[2]

    # creating a dictionary of the model decisions (hyperplane distance) per layer 
    img_decision_dic = {}
    for img, decision, true_label, predicted_label  in zip(imgs, decisions, true_labels, predicted_labels):
        if int(true_label) != int(predicted_label):
            img_decision_dic[img] = -abs(float(decision)) # missclassified examples get a negative value
        else:
            img_decision_dic[img] = abs(float(decision)) # correctly classified exampled get a positive value

    total_avg_human_scores = []
    model_decisions = []

    # populating total_avg_human_scores per image and model_decisions per image
    for img in img_decision_dic.keys():
        human_scores = scores_by_img[img]
        avg_human_scores = np.mean(np.array(human_scores))
        total_avg_human_scores.append((avg_human_scores*100))
        model_decisions.append(img_decision_dic[img])

    # computing correlations
    corrl, p_val = stats.spearmanr(total_avg_human_scores, model_decisions)
    print "model accuracy: {0}".format(accuracy_score(true_labels,predicted_labels))
    correlation_msg = "correlation: {0}".format(corrl)
    print correlation_msg
    print "p value: {0}".format(p_val)

    colors = ['blue','green']
    colors = ['blue' if int(label) == 1 else 'green' for label in true_labels]

    # plotting
    plt.scatter(total_avg_human_scores, model_decisions, c=colors,alpha=0.4)
    plt.scatter([],[],color='blue',label='Animal',alpha=0.4)
    plt.scatter([],[],color='green',label='Non-Animal',alpha=0.4)
    plt.xlim(-10,100)
    plt.ylim(-2, 4)
    plt.axvline(x=0, color ='black')
    plt.axhline(y=0, color ='black')
    plt.axvline(x=50, color='red', linestyle='--',label='Chance')
    title = 'Fitting Human to Vgg16 ' + str(layer_file.split('/')[-1].split('.')[0][:-8]) + ' Accuracy'
    plt.title(title)
    correlation_and_pvalue = 'Correlation: {0:.3g} (Spearman\'s rho), p-value: {1:.3g}'.format(corrl, p_val)
    plt.xlabel('Human Accuracy (%)' +'\n' +'\n' + correlation_and_pvalue)
    plt.ylabel('Distance from Hyperplane')
    legend = plt.legend(loc=2)
    frame = legend.get_frame()
    frame.set_color('white')
    # plt.show()

    results_txt = layer_file[:-11] + 'correlation_results.txt'
    with open(results_txt, 'wt') as f:
        f.write(correlation_msg +'\n')

def spearmans_rho_per_layer_nips(scores_by_img, layer_name):
    """
    Computes The Spearman's rho correlation between Distance from the Hyperplane and Human Accuracy
    plots the resulting correlation for the original NIPS results with the non-finetuned Vgg16 model
    Input: scores_by_img - a dictionary of scores per image
           layer_name - the layer name to evalute the correlation for
    Output: None
    """
    sets = ['turk_30.npz', 'turk_31.npz', 'turk_32.npz', 'turk_33.npz', 'turk_34.npz']
    # data is located on g17 machine
    directory = '/media/storage/charlie/replicate_results/sven_predictions/predictions/VGG16_' + layer_name + 'ex/svm_0-15/'
    true_labels = []
    predicted_labels = []
    decisions = []
    imgs = []
    for s in sets:
        array = np.load(directory + s)
        predictions_list = array['pred_labels'].tolist()
        true_labels_list = array['true_labels'].tolist()
        decisions_list = array['hyper_dist'].tolist()
        img_list = array['source_filenames'].tolist()

        predicted_labels += predictions_list
        true_labels += true_labels_list
        decisions += decisions_list
        imgs += img_list

    # creating a dictionary of the model decisions (hyperplane distance) per layer 
    img_decision_dic = {}
    for img, decision, true_label, predicted_label  in zip(imgs, decisions, true_labels, predicted_labels):
        if int(true_label) != int(predicted_label):
            img_decision_dic[img] = -abs(float(decision)) # missclassified examples get a negative value
        else:
            img_decision_dic[img] = abs(float(decision)) # correctly classified exampled get a positive value

    total_avg_human_scores = []
    model_decisions = []
    # populating total_avg_human_scores per image and model_decisions per image
    for img in img_decision_dic.keys():
        human_scores = scores_by_img[img]
        avg_human_scores = np.mean(np.array(human_scores))
        total_avg_human_scores.append((avg_human_scores*100))
        model_decisions.append(img_decision_dic[img])

    # computing correlations
    corrl, p_val = stats.spearmanr(total_avg_human_scores, model_decisions)
    print "model accuracy: {0}".format(accuracy_score(true_labels,predicted_labels))
    print "correlation: {0}".format(corrl)
    print "p value: {0}".format(p_val)

    colors = ['blue','green']
    colors = ['blue' if int(label) == 1 else 'green' for label in true_labels]

    # plotting
    plt.scatter(total_avg_human_scores, model_decisions, c=colors,alpha=0.4)
    plt.scatter([],[],color='blue',label='Animal',alpha=0.4)
    plt.scatter([],[],color='green',label='Non-Animal',alpha=0.4)
    plt.xlim(-10,100)
    plt.ylim(-2, 4)
    plt.axvline(x=0, color ='black')
    plt.axhline(y=0, color ='black')
    plt.axvline(x=50, color='red', linestyle='--',label='Chance')
    title = 'Fitting Human to Vgg16 ' + str(layer_name) + ' Accuracy'
    plt.title(title)
    correlation_and_pvalue = 'Correlation: {0:.3g} (Spearman\'s rho), p-value: {1:.3g}'.format(corrl, p_val)
    plt.xlabel('Human Accuracy (%)' +'\n' +'\n' + correlation_and_pvalue)
    plt.ylabel('Distance from Hyperplane')
    legend = plt.legend(loc=2)
    frame = legend.get_frame()
    frame.set_color('white')
    plt.show()

    

if __name__ == '__main__':

    # set0 = ('/media/data_cifs/nsf_levels/Results/set0.db') control set
    set30 = ('/media/data_cifs/nsf_levels/Results/set30.db')
    set31 = ('/media/data_cifs/nsf_levels/Results/set31.db')
    set32 = ('/media/data_cifs/nsf_levels/Results/set32.db')
    set33 = ('/media/data_cifs/nsf_levels/Results/set33.db')
    set34 = ('/media/data_cifs/nsf_levels/Results/set34.db')

    scores = []

    scores_by_img = {}
    scores_by_participant = {}

    # append_scores(set0, scores_by_participant, scores_by_img, scores)
    append_scores(set30, scores_by_participant, scores_by_img, scores)
    append_scores(set31, scores_by_participant, scores_by_img, scores)
    append_scores(set32, scores_by_participant, scores_by_img, scores)
    append_scores(set33, scores_by_participant, scores_by_img, scores)
    append_scores(set34, scores_by_participant, scores_by_img, scores)    

    parser = argparse.ArgumentParser()
    parser.add_argument('layer')
    parser.add_argument('results_dir')
    args = parser.parse_args()
    layer_file = args.results_dir + "/" + args.layer + "_results.npy"
    
    spearmans_rho_per_layer(scores_by_img, layer_file)
    # spearmans_rho_per_layer_nips(scores_by_img, args.layer)

    # calculating human level accuracy 
    accuracies = []
    for v in scores_by_participant.values():
        accuracy = sum(v)/len(v)
        accuracies.append(accuracy)

    print "Human level accuracy mean:", sum(accuracies)/len(accuracies)
    print "Human level accuracy std", np.array(accuracies).std()


    


   

    




    