import sklearn.metrics

sklearn.metrics.recall_score(y_true, y_pred, pos_label=1)
sklearn.metrics.precision_score(y_true, y_pred, pos_label=1)

sklearn.metrics.roc_auc_score(y_true, (-1) * y_score)
sklearn.metrics.average_precision_score(y_true, (-1) * y_score, pos_label=1)
