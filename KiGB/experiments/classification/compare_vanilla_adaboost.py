import numpy as np
import pandas as pd
import logging
import sys
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from core.scikit.gradient_boosting import GradientBoostingClassifier
    from core.scikit.skigb import SKiGB
    from experiments.classification.setting import *
except ImportError:
    sys.path.append('./')
    from core.scikit.gradient_boosting import GradientBoostingClassifier
    from core.scikit.skigb import SKiGB
    from experiments.classification.setting import *

import xgboost as xgb

logging.basicConfig(format='%(message)s', level=logging.INFO)

data_list = ['adult', 'australia', 'car', 'cleveland', 'ljubljana']
fold_size = 5
num_epochs = 5  # Total number of runs

def get_error(clf_model, X, y):
    y_pred = clf_model.predict(X)
    return accuracy_score(y, y_pred)

for dataset in data_list:
    dirName = './datasets/classification/' + dataset
    test_data = pd.read_csv(dirName + '/test.csv')
    target = data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    # Store all accuracy values across epochs × folds
    kigb_scores = []
    gb_scores = []
    xgb_scores = []
    ada_scores = []

    for epoch in range(num_epochs):
        for fold in range(fold_size):
            train_data = pd.read_csv(f"{dirName}/train_{fold}.csv")
            X_train = train_data.drop(target, axis=1)
            y_train = train_data[target]

            # SKiGB
            skigb = SKiGB(criterion='squared_error',
                          n_estimators=30,
                          max_depth=14,
                          learning_rate=0.1,
                          loss='deviance',
                          random_state=epoch,
                          advice=advice,
                          lamda=data_penalty[dataset],
                          epsilon=data_margin[dataset])
            skigb.fit(X_train, y_train)
            kigb_scores.append(get_error(skigb.kigb, X_test, y_test))

            # Scikit-GB
            clf = GradientBoostingClassifier(criterion='squared_error',
                                             n_estimators=30,
                                             max_depth=14,
                                             learning_rate=0.1,
                                             loss='deviance',
                                             random_state=epoch)
            clf.fit(X_train, y_train)
            gb_scores.append(get_error(clf, X_test, y_test))

            # XGBoost
            xgb_clf = xgb.XGBClassifier(n_estimators=30,
                                        max_depth=14,
                                        learning_rate=0.1,
                                        use_label_encoder=False,
                                        eval_metric='logloss',
                                        verbosity=0,
                                        random_state=epoch)
            xgb_clf.fit(X_train, y_train)
            xgb_scores.append(get_error(xgb_clf, X_test, y_test))

            # AdaBoost
            ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=14),
                                     n_estimators=30,
                                     learning_rate=0.1,
                                     random_state=epoch)
            ada.fit(X_train, y_train)
            ada_scores.append(get_error(ada, X_test, y_test))

    # Aggregate metrics
    kigb_accuracy = np.mean(kigb_scores)
    gb_accuracy = np.mean(gb_scores)
    xgb_accuracy = np.mean(xgb_scores)
    ada_accuracy = np.mean(ada_scores)

    kigb_std = np.std(kigb_scores)
    gb_std = np.std(gb_scores)
    xgb_std = np.std(xgb_scores)
    ada_std = np.std(ada_scores)

    # Paired t-tests (across all 25 accuracy values)
    ttest_kigb_gb = ttest_rel(kigb_scores, gb_scores)
    ttest_kigb_xgb = ttest_rel(kigb_scores, xgb_scores)
    ttest_kigb_ada = ttest_rel(kigb_scores, ada_scores)

    # Report
    logging.info(f"\nDataset: '{dataset}'")
    logging.info(f"  SKiGB Accuracy:    {kigb_accuracy:.3f}")
    logging.info(f"  Scikit-GB Accuracy:{gb_accuracy:.3f}")
    logging.info(f"  XGBoost Accuracy:  {xgb_accuracy:.3f}")
    logging.info(f"  AdaBoost Accuracy: {ada_accuracy:.3f}")
    logging.info(f"  T-test (SKiGB vs SGB):  p = {ttest_kigb_gb.pvalue:.4f}")
    logging.info(f"  T-test (SKiGB vs XGB):  p = {ttest_kigb_xgb.pvalue:.4f}")
    logging.info(f"  T-test (SKiGB vs Ada):  p = {ttest_kigb_ada.pvalue:.4f}")
