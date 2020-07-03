## Detecting Attacks on a Robotic Vehicle

### Overview

The task of detecting attacks on a robotic vehicle is approached as a binary classification problem. Given the vehicle's state at time `t` (and possibly previous states), the task is to classify whether the vehicle is under attack (Anomaly) or not (Normal). 

Although it is an unsupervised classification task, it is possible to approach it using semi-supervised and/or supervised learning. This can be achieved by labeling the training examples with the most confident predictions of the unsupervised model. Using these "noisy" labels, we can train a semi-supervised or even a supervised model. The idea is that, although the model will generate several wrong labels, learning a decision function can be quite robust to wrongly labeled data, since most of the signal is still present in the data. We describe the above process in three steps:

1) Unsupervised: We start with an unsupervised model without any knowledge of the class labels. For that, we train an Isolation Forest model.
2) Semi-supervised: We label the top-k confident examples predicted as "Normal" from the unsupervised model. Then, we train a novelty detection (semi-supervised) algorithm, namely, One-Class SVM, on this noisy training set in order to learn the distribution of the "Normal" class. For a new (unseen) example, the model predicts whether it is generated from the training distribution or not, i.e. classifies the example as "Normal" or "Anomaly".
3) Supervised: This is similar to 2), but we obtain the top-k confident examples from both classes ("Normal" and "Anomaly"). Finally, we train a supervised model, such as a Random Forest, on the fully labeled training set.    


### How to Use

Clone or download this repository.
```
$ git clone https://github.com/micts/anomaly-detection-robotic-vehicle.git
```

#### Task 2
We deploy the trained models as a REST API using Docker and Flask.

Build a docker image from the Dockerfile
```
$ docker build -t <name_for_image>:<tag> .
```
For example,
```
$ docker build -t ad_image_test:v1 .
```
The above command will install all required dependencies. Additionally, by building the image, we train the classification models. These models will be saved and made available in every container initialized from the built image.   

To run the server, we use
```
$ docker run -it -p 5000:5000 ad_image_test:v1 python3 api.py -mn model_name
```
where we subsitute `model_name` with one of the following: `isolation_forest` (unsupervised), `one-class_svm` (semi-supervised), or `random_forest` (supervised). It is also possible to train these models using additional lagged variables by specifying the -lv option
```
$ docker run -it -p 5000:5000 ad_image_test:v1 python3 api.py -mn model_name -lv
```
For example, we run the server and perform inference using the trained Random Forest model with lagged variables 
```
$ docker run -it -p 5000:5000 ad_image_test:v1 python3 api.py -mn random_forest -lv
```
We can issue a request by running the shell script `request_lag_models.sh` for models with lagged variables or `request.sh` for models with no lagged variables.  By running the following command, our model should output its prediction, either "Normal" - (0) or "Anomaly" - (1)
```
./request_lag_models.sh
```

