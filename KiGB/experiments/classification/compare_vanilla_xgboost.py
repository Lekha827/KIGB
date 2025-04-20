# Import libraries necessary for this project
import numpy as np
import pandas as pd
import logging
import sys
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score

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

# Use this list to test multiple datasets
data_list = ['adult', 'australia', 'ljubljana']
# data_list = ['ljubljana']  # For testing a single dataset

def get_error(clf_model, X, y):
    y_pred = clf_model.predict(X)
    return accuracy_score(y, y_pred)

for dataset in data_list:
    dirName = './datasets/classification/' + dataset
    test_data = pd.read_csv(f"{dirName}/test.csv")
    target = data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    fold_size = 5
    kigb_score = np.zeros(fold_size)
    gb_score = np.zeros(fold_size)
    xgb_score = np.zeros(fold_size)

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
                      random_state=12,
                      advice=advice,
                      lamda=5,
                      epsilon=-0.3)
        skigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(skigb.kigb, X_test, y_test)

        # Scikit-GB
        clf = GradientBoostingClassifier(criterion='squared_error',
                                         n_estimators=30,
                                         max_depth=14,
                                         learning_rate=0.1,
                                         loss='deviance',
                                         random_state=12)
        clf.fit(X_train, y_train)
        gb_score[fold] = get_error(clf, X_test, y_test)

        # XGBoost
        xgb_clf = xgb.XGBClassifier(n_estimators=30,
                                    max_depth=14,
                                    learning_rate=0.1,
                                    use_label_encoder=False,
                                    eval_metric='logloss',
                                    verbosity=0,
                                    random_state=12)
        xgb_clf.fit(X_train, y_train)
        xgb_score[fold] = get_error(xgb_clf, X_test, y_test)

    # Averages
    kigb_acc = np.mean(kigb_score)
    gb_acc = np.mean(gb_score)
    xgb_acc = np.mean(xgb_score)

    kigb_std = np.std(kigb_score)
    gb_std = np.std(gb_score)
    xgb_std = np.std(xgb_score)

    # Paired t-tests
    ttest_kigb_xgb = ttest_rel(xgb_score, kigb_score)
    ttest_kigb_gb = ttest_rel(kigb_score, gb_score)

    # Report
    logging.info(f"\nDataset: '{dataset}' SKiGB Accuracy: {kigb_acc:.3f};  SGB Accuracy: {gb_acc:.3f}; XGBoost Accuracy: {xgb_acc:.3f}")
    logging.info(f"  T-test (XGBoost vs SKiGB):     p-value = {ttest_kigb_xgb.pvalue:.10f}")
    logging.info(f"  T-test (SKiGB vs Scikit-GB):   p-value = {ttest_kigb_gb.pvalue:.10f}")
