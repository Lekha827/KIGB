# Import libraries necessary for this project
import numpy as np
import pandas as pd
import logging
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

import xgboost as xgb  # NEW

logging.basicConfig(format='%(message)s', level=logging.INFO)

data_list = ['abalone','autompg','autoprice','boston','california','cpu','crime','redwine','whitewine','windsor']
# data_list = ['abalone']

def get_error(reg_model, X, y):
    y_pred = reg_model.predict(X)
    return mean_squared_error(y, y_pred)

for dataset in data_list:
    dirName = './datasets/regression/' + dataset
    test_data = pd.read_csv(dirName + '/test.csv')
    target = data_target[dataset]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]
    advice = np.array(data_advice[dataset].split(','), dtype=int)

    fold_size = 5
    kigb_score = np.zeros((fold_size), dtype=np.float64)
    gb_score = np.zeros((fold_size), dtype=np.float64)
    xgb_score = np.zeros((fold_size), dtype=np.float64)  # NEW

    for fold in range(fold_size):
        train_data = pd.read_csv(dirName + '/train_' + str(fold) + '.csv')
        X_train = train_data.drop(target, axis=1)
        y_train = train_data[target]

        # SKiGB
        skigb = SKiGB(criterion='squared_error',
                      n_estimators=30,
                      max_depth=10,
                      learning_rate=0.1,
                      loss='ls',
                      random_state=12,
                      advice=advice,
                      lamda=5,
                      epsilon=-0.3
                      )
        skigb.fit(X_train, y_train)
        kigb_score[fold] = get_error(skigb.kigb, X_test, y_test)

        # Scikit-GB
        reg = GradientBoostingRegressor(criterion='squared_error',
                                        n_estimators=30,
                                        max_depth=10,
                                        learning_rate=0.1,
                                        loss='ls',
                                        random_state=12
                                        )
        reg.fit(X_train, y_train)
        gb_score[fold] = get_error(reg, X_test, y_test)

        # XGBoost
        xgb_reg = xgb.XGBRegressor(n_estimators=30,
                                   max_depth=10,
                                   learning_rate=0.1,
                                   objective='reg:squarederror',
                                   verbosity=0,
                                   random_state=12)
        xgb_reg.fit(X_train, y_train)
        xgb_score[fold] =  get_error(xgb_reg, X_test, y_test)

    # Averages
    kigb_mse = np.mean(kigb_score)
    gb_mse = np.mean(gb_score)
    xgb_mse = np.mean(xgb_score)

    kigb_std = np.std(kigb_score)
    gb_std = np.std(gb_score)
    xgb_std = np.std(xgb_score)

    # Paired t-tests
    ttest_kigb_xgb = ttest_rel(kigb_score, xgb_score)
    ttest_kigb_gb = ttest_rel(kigb_score, gb_score)

    # Report
    logging.info(f"\nDataset: '{dataset}' SKiGB MSE:  {kigb_mse:.3f};  SGB MSE:  {gb_mse:.3f}; XGBoost MSE:   {xgb_mse:.3f}")
    logging.info(f"  T-test (SKiGB vs XGBoost ):     p-value = {ttest_kigb_xgb.pvalue:.10f}")
    logging.info(f"  T-test (SKiGB vs SGB): p-value = {ttest_kigb_gb.pvalue:.10f}")
