import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, roc_auc_score, average_precision_score

import vis_utils

def relabel_pred(pred):
    """
    Re-label predictions to be compatible with flag (target).
    1 --> 0
    -1 --> 1
    """
    return ((-0.5) * (pred - 1)).astype(np.int)

def get_train_inds(scores, pred, labels=[0, 1]):

    """
    Returns training set indices of top-k "normal" (0) or "anomalous" (1) predictions produced by the unsupervised model.
    The returned instances are labeled (based on the prediction) to train the semi-supervised and supervised model.
    """

    # index scores
    inds = np.arange(len(scores))
    scores = np.stack((scores, inds)).T

    # for both semi-supervised and supervised case
    normal_obs = scores[pred == 0, :] # predicted as "normal"
    normal_obs_ordered = normal_obs[normal_obs[:, 0].argsort()[::-1]] # sort w.r.t score
    topk_normal_obs = normal_obs_ordered[:1670] # extract top k # 1650
    inds_to_keep = np.in1d(inds, topk_normal_obs[:, 1]) # indices in top-k

    # only for supervised case
    if 1 in labels:
        anomalous_obs = scores[pred == 1, :] # predicted as "anomaly"
        anomalous_obs_ordered = anomalous_obs[anomalous_obs[:, 0].argsort()] # sort w.r.t score
        topk_anomalous_obs = anomalous_obs_ordered[:1750] # extract top k # 1725
        inds_to_keep = np.vstack((inds_to_keep, np.in1d(inds, topk_anomalous_obs[:, 1]))) # indices in top-k

    return inds_to_keep

def create_lag_variables(data):
    """
    Creates lag versions of two time steps (t-1, t-2) for each feature in the dataset.
    """

    for n_column in range(1, len(data.columns)):
        column = data.columns[n_column]
        if column == 'flag':
            continue
        data[column + '(t-1)'] = data[column].shift(1)
        data[column + '(t-2)'] = data[column].shift(2)
    return data.iloc[2:]

def construct_features(data, is_lag):
    """
    Constructs two new features and lag versions for each feature.
    """
    data['Volts'] = data['Watts'] / data['Amps'] # Interaction between Watts and Amps, as Volts = Watts / Amps
    data['R/T(xKBTot)'] = (data['RxKBTot'] + 1) / (data['TxKBTot'] + 1) # Ratio of received to transmitted network traffic rate
    if is_lag:
        data = create_lag_variables(data)
    return data

def scale_data(x_train, x_test):
    """
    Scale data to zero mean and unit variance.
    """

    standard_scaler = StandardScaler()
    x_train = standard_scaler.fit_transform(x_train)
    x_test = standard_scaler.transform(x_test)
    return x_train, x_test, standard_scaler

def model_evaluation(name, y_true, y_pred, y_score):
    """
    Evaluate model w.r.t. 1) Recall
                          2) Precision
                          3) Area under the Receiver Operation Characteristic (ROC) curve - AUROC
                          4) Area under the Precision-Recall curve (Average Precision) - AP
    """
    recall = recall_score(y_true, y_pred, pos_label=1)
    precision = precision_score(y_true, y_pred, pos_label=1)

    if name == "Random Forest (Supervised)":
        roc_auc = roc_auc_score(y_true, y_score[:, 1])
        ap_auc = average_precision_score(y_true, y_score[:, 1], pos_label=1)
    else:
        roc_auc = roc_auc_score(y_true, (-1) * y_score)
        ap_auc = average_precision_score(y_true, (-1) * y_score, pos_label=1)

    return recall, precision, roc_auc, ap_auc

def extract_feature_importances(model, features):
    """
    Extract feature importances produced by random forest, and sort the in descending order.
    """
    features = features.tolist()
    f_fi = [(feature, model.feature_importances_[idx]) for idx, feature in enumerate(features)]
    f_fi.sort(key = lambda x: x[1], reverse=True)
    return f_fi

def save_feature_importances(f_fi, is_lag):
    if not os.path.exists('feature_importances/'):
        os.mkdir('feature_importances/')
    if is_lag:
        f_fi_name = 'feature_importances_lag'
    else:
        f_fi_name = 'feature_importances'
    with open('feature_importances/' + f_fi_name + '.pkl', 'wb') as f:
        pickle.dump(f_fi, f)
    vis_utils.plot_feature_importances(f_fi, is_lag)

def print_feature_importances(f_fi):
    for i in f_fi:
        print(i[0] + ':', i[1])

def append_results(name, results, model_results):
    metrics = ['Recall', 'Precision', 'Area Under ROC (AUROC)', 'Area Under PRC (Average Precision)']
    if 'train' not in results:
        results['train'] = {}
    if 'test' not in results:
        results['test'] = {}
    for split in ['train', 'test']:
        results[split][name] = {}
        for idx, metric in enumerate(metrics):
            if split == 'train':
                results[split][name][metric] = model_results[0][idx]
            else:
                results[split][name][metric] = model_results[1][idx]
    return results

def save_model(model, model_name, is_lag):
    if is_lag:
        model_name = '_'.join(model_name.lower().split()) + '_lag'
    else:
        model_name = '_'.join(model_name.lower().split())
    if not os.path.exists('models/'):
        os.mkdir('models/')
    with open('models/' + model_name + '.pkl', 'wb') as f:
        pickle.dump(model, f)

def save_transform(scaler, is_lag):
    if is_lag:
        transform_name = 'transform' + '_lag'
    else:
        transform_name = 'transform'
    if not os.path.exists('transforms/'):
        os.mkdir('transforms/')
    with open('transforms/' + transform_name + '.pkl', 'wb') as f:
        pickle.dump(scaler, f)

def save_results(results, is_lag):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if is_lag:
        results_name = 'results_lag'
    else:
        results_name = 'results'
    with open('results/' + results_name + '.pkl', 'wb') as f:
        pickle.dump(results, f)

def print_results(results):
    print('\n\n')
    print('Results', 8 * '-', sep='\n')
    for split in results:
        if split == 'train':
            print('', 'Training Set', 12 * '-', sep='\n')
        else:
            print('', 'Test Set', 8 * '-', sep='\n')
        for name in results[split]:
            print('', name, 10 * '-', sep='\n')
            for metric, value in results[split][name].items():
                print(metric, round(value, 3))
