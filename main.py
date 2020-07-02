import argparse
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier

import utils

parser = argparse.ArgumentParser()
parser.add_argument('-dp', '--data_path', help='Path to .csv file')
parser.add_argument('-lv', '--lag_variables', action='store_true', help='Create a lag version of two time steps for each variable')
parser.add_argument('-nv', '--no_verbose', action='store_true', help='No verbose for results')
parser.add_argument('-sr', '--save_results', action='store_true', help='Save results in a txt file')
parser.add_argument('-sm', '--save_models', action='store_true', help='Save trained models')
parser.add_argument('-fi', '--feature_importances', action='store_true', help='Extract feature importances from random forest')

args = parser.parse_args()
data_path = args.data_path
is_lag = args.lag_variables
is_verbose = not args.no_verbose
save_results = args.save_results
save_models = args.save_models
feature_importances = args.feature_importances

data = pd.read_csv(data_path)

data, features = utils.construct_features(data, is_lag)
print(data.columns[features])
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, features], # 26
                                                    data.iloc[:, 9],
                                                    test_size=0.33,
                                                    random_state=151) # 42




# if lag_variables:
#     data['Volts'] = data['Watts'] / data['Amps']
#     data['R/T(xKBTot)'] = (data['RxKBTot'] + 1) / (data['TxKBTot'] + 1)
#     data = utils.create_lag_variables(data)
#     print(data.columns[pd.np.r_[1, 2, 3, 4, 7, 8, 10:20, 24:32]])
#     x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, pd.np.r_[1, 2, 3, 4, 7, 8, 10:20, 24:32]], # 26
#                                                         data.iloc[:, 9],
#                                                         test_size=0.33,
#                                                         random_state=151) # 42
#     #print(data.columns)
# else:
#     data['Volts'] = data['Watts'] / data['Amps']
#     #data['RxKBTot'] = np.log(data['RxKBTot'] + 1)
#     #data['TxKBTot'] = np.log(data['TxKBTot'] + 1)
#     data['R/T(xKBTot)'] = (data['RxKBTot'] + 1) / (data['TxKBTot'] + 1)
#     #data['R/T(xKBTot)'] = np.log(data['RxKBTot'] + 1) - np.log(data['TxKBTot'] + 1)
#     print(data.columns[[1, 2, 3, 4, 7, 8, 10, 11]])
#     x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, [1, 2, 3, 4, 7, 8, 10, 11]], # 1:9
#                                                         data.iloc[:, 9],
#                                                         test_size=0.33,
#                                                         random_state=151)

x_train, x_test = utils.scale_data(x_train, x_test)
x_train_ = np.copy(x_train)
y_train_ = np.copy(y_train)

models = [
    ('Isolation Forest (Unsupervised)', IsolationForest(n_estimators=1000,
                                          contamination=.5,
                                          max_features=1,
                                          max_samples=1000,
                                          random_state=0)),
    ('One-Class SVM (Semi-Supervised)', OneClassSVM(kernel='linear',
                                                    nu=0.2)),
    ('Random Forest (Supervised)', RandomForestClassifier(n_estimators=1000,
                                                    max_features=1,
                                                    max_samples=1000,
                                                    random_state=0))]

results = {}
for name, model in models:

    x_train = np.copy(x_train_)
    y_train = np.copy(y_train_)

    print('\n', name, 10 * '-', sep='\n')
    if name == 'One-Class SVM (Semi-Supervised)':

        print('Labeling training set based on top-k unsupervised predictions from \'Normal\' class...')
        time.sleep(2)
        inds = utils.get_train_inds(y_score_train, y_pred_train, labels=[0])
        assert inds.ndim == 1

        # training set comprised of top-k scored examples from "normal" class
        x_train = x_train[inds, :]
        y_train = y_train[inds]

    elif name == 'Random Forest (Supervised)':

        print('Labeling training set based on top-k unsupervised predictions from both classes...')
        time.sleep(2)
        y_score_train = np.copy(y_score_train_)
        y_pred_train = np.copy(y_pred_train_)

        inds = utils.get_train_inds(y_score_train, y_pred_train, labels=[0, 1])
        assert inds.ndim == 2

        # training set comprised of top-k scored examples from both classes
        x_train = np.vstack((x_train[inds[0], :], x_train[inds[1], :]))
        y_train = np.hstack((y_pred_train[inds[0]], y_pred_train[inds[1]]))

    print('Fitting ', name.rsplit(' ', 1)[0], '...', sep='')
    time.sleep(1)
    if name == 'Random Forest (Supervised)':
        model.fit(x_train, y_train)
        if feature_importances:
            f_fi = utils.extract_feature_importances(model, data.columns[features])
    else:
        model.fit(x_train)

    model_name = name.rsplit(' ', 1)[0]
    if save_models:
        utils.save_model(model, model_name, is_lag)

    if name == 'Random Forest (Supervised)':
        y_score_train = model.predict_proba(x_train)
        y_score_test = model.predict_proba(x_test)
    else:
        y_score_train = model.decision_function(x_train)
        y_score_test = model.decision_function(x_test)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    if name == 'Isolation Forest (Unsupervised)' or name == 'One-Class SVM (Semi-Supervised)':
        y_pred_train = utils.relabel_pred(y_pred_train)
        y_pred_test = utils.relabel_pred(y_pred_test)

    if name == 'Random Forest (Supervised)':
        #y_train = np.copy(y_train_)
        y_train = np.hstack((y_train_[inds[0]], y_train_[inds[1]]))

    print('Evaluating ', name.rsplit(' ', 1)[0], '...', sep='')
    time.sleep(2)

    train_results = utils.model_evaluation(name, y_train, y_pred_train, y_score_train)
    test_results = utils.model_evaluation(name, y_test, y_pred_test, y_score_test)
    model_results = [train_results, test_results]

    results = utils.append_results(name, results, model_results)

    if name == 'Isolation Forest (Unsupervised)':
        y_score_train_ = np.copy(y_score_train)
        y_pred_train_ = np.copy(y_pred_train)

if save_results:
    utils.save_results(results, is_lag)

if is_verbose:
    time.sleep(1)
    utils.print_results(results)

if feature_importances:
    print('\n\n')
    print('Ordered feature importances (Random Forest)', 8 * '-', sep='\n')
    utils.print_feature_importances(f_fi)
    utils.save_feature_importances(f_fi, is_lag)

print(features)
