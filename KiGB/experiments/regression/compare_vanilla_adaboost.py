import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import ttest_rel
from sklearn.metrics import mean_squared_error
import sys

try:
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from core.scikit.skigb import SKiGB
    from experiments.regression.setting import *
except ImportError:
    sys.path.append('./')
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from core.scikit.skigb import SKiGB
    from experiments.regression.setting import *

import xgboost as xgb

logging.basicConfig(format='%(message)s', level=logging.INFO)

data_list = ['abalone', 'autompg', 'autoprice', 'boston', 'california',
             'cpu', 'crime', 'redwine', 'whitewine', 'windsor']

def get_error(reg_model, X, y):
    y_pred = reg_model.predict(X)
    return mean_squared_error(y, y_pred)

num_epochs = 5
fold_size = 5

for dataset in data_list:
    dirName = './datasets/regression/' + dataset
    test_data = pd.read_csv(dirName + '/test.csv')
    target = data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    # Collect all MSEs across folds and epochs
    kigb_scores = []
    gb_scores = []
    xgb_scores = []
    ada_scores = []

    for epoch in range(num_epochs):
        for fold in range(fold_size):
            train_data = pd.read_csv(dirName + f'/train_{fold}.csv')
            X_train = train_data.drop(target, axis=1)
            y_train = train_data[target]

            # SKiGB
            skigb = SKiGB(criterion='squared_error',
                          n_estimators=30,
                          max_depth=10,
                          learning_rate=0.1,
                          loss='ls',
                          random_state=epoch,
                          advice=advice,
                          lamda=data_penalty[dataset],
                          epsilon=data_margin[dataset])
            skigb.fit(X_train, y_train)
            kigb_scores.append(get_error(skigb.kigb, X_test, y_test))

            # Scikit-GB
            reg = GradientBoostingRegressor(criterion='squared_error',
                                            n_estimators=30,
                                            max_depth=10,
                                            learning_rate=0.1,
                                            loss='ls',
                                            random_state=epoch)
            reg.fit(X_train, y_train)
            gb_scores.append(get_error(reg, X_test, y_test))

            # XGBoost
            xgb_reg = xgb.XGBRegressor(n_estimators=30,
                                       max_depth=10,
                                       learning_rate=0.1,
                                       objective='reg:squarederror',
                                       verbosity=0,
                                       random_state=epoch)
            xgb_reg.fit(X_train, y_train)
            xgb_scores.append(get_error(xgb_reg, X_test, y_test))

            # AdaBoost
            ada = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=10),
                                    n_estimators=30,
                                    learning_rate=0.1,
                                    random_state=epoch)
            ada.fit(X_train, y_train)
            ada_scores.append(get_error(ada, X_test, y_test))

    # Averages
    kigb_mse = np.mean(kigb_scores)
    gb_mse = np.mean(gb_scores)
    xgb_mse = np.mean(xgb_scores)
    ada_mse = np.mean(ada_scores)

    kigb_std = np.std(kigb_scores)
    gb_std = np.std(gb_scores)
    xgb_std = np.std(xgb_scores)
    ada_std = np.std(ada_scores)

    # T-tests
    ttest_kigb_gb = ttest_rel(kigb_scores, gb_scores)
    ttest_kigb_xgb = ttest_rel(kigb_scores, xgb_scores)
    ttest_kigb_ada = ttest_rel(kigb_scores, ada_scores)

    logging.info(f"\nDataset: '{dataset}'")
    logging.info(f"  SKiGB MSE:    {kigb_mse:.3f}")
    logging.info(f"  SGB MSE:      {gb_mse:.3f}")
    logging.info(f"  XGBoost MSE:  {xgb_mse:.3f}")
    logging.info(f"  AdaBoost MSE: {ada_mse:.3f}")
    logging.info(f"T-test (SKiGB vs SGB):      p-value = {ttest_kigb_gb.pvalue:.4f}")
    logging.info(f"T-test (SKiGB vs XGBoost):  p-value = {ttest_kigb_xgb.pvalue:.4f}")
    logging.info(f"T-test (SKiGB vs AdaBoost): p-value = {ttest_kigb_ada.pvalue:.4f}")
