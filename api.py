import os
import argparse
import pickle

from flask import Flask
from flask_restful import Resource, Api, reqparse
from joblib import load
import numpy as np

import utils

parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model_name', help='Model to perform inference with. Possible values are \'isolation_forest\', \'one-class_svm\', \'random_forest\'.')
parser.add_argument('-lv', '--lag_variables', action='store_true', help='Whether to include lag variables in the model.')

args = parser.parse_args()
model_name = args.model_name
is_lag = args.lag_variables

assert model_name == 'isolation_forest' or \
       model_name == 'one-class_svm' or \
       model_name == 'random_forest', \
       'Please provide a valid model name. Possible values are \'isolation_forest\', \'one-class_svm\', \'random_forest\'.'

if is_lag:
    model_dir = 'models/' + model_name + '_lag' + '.pkl'
    transform_dir = 'transforms/transform_lag.pkl'
else:
    model_dir = 'models/' + model_name + '.pkl'
    transform_dir = 'transforms/transform.pkl'

print('\n', 'Loading model from', model_dir)
with open(model_dir, 'rb') as f:
    model = pickle.load(f)

print('\n', 'Loading transform from', transform_dir, '\n')
with open(transform_dir, 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)
api = Api(app)

if is_lag:
    features = ['CPU', 'RxKBTot', 'TxKBTot', 'WriteKBTot',
                'RMS', 'diff_encoder_l','Volts', 'R/T(xKBTot)',
                'CPU(t-1)', 'CPU(t-2)', 'RxKBTot(t-1)', 'RxKBTot(t-2)',
                'TxKBTot(t-1)', 'TxKBTot(t-2)', 'WriteKBTot(t-1)', 'WriteKBTot(t-2)',
                'RMS(t-1)', 'RMS(t-2)', 'diff_encoder_l(t-1)', 'diff_encoder_l(t-2)',
                'Volts(t-1)', 'Volts(t-2)', 'R/T(xKBTot)(t-1)', 'R/T(xKBTot)(t-2)']
else:
    features = ['CPU', 'RxKBTot', 'TxKBTot', 'WriteKBTot',
                'RMS', 'diff_encoder_l', 'Volts', 'R/T(xKBTot)']

parser = reqparse.RequestParser()
for feature in features:
    parser.add_argument(feature, type=float, location='json')

class Prediction(Resource):

    def post(self):
        args = parser.parse_args()
        x = np.array([args[f] for f in features]).reshape(1, -1)
        x = scaler.transform(x)
        y_pred = model.predict(x)
        if model_name == 'isolation_forest' or model_name == 'one-class_svm':
            y_pred = utils.relabel_pred(y_pred)
        if y_pred == 0:
            prediction = 'Normal (0)'
        elif y_pred == 1:
            prediction = 'Anomaly (1)'
        return {'Prediction': prediction}

api.add_resource(Prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
