import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys

try:
    from core.scikit.skigb import SKiGB
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from experiments.regression.setting import *
except ImportError:
    sys.path.append('./')
    from core.scikit.skigb import SKiGB
    from core.scikit.gradient_boosting import GradientBoostingRegressor
    from experiments.regression.setting import *

import xgboost as xgb

# PARAMETERS
dataset = 'crime'
epsilon_values = [-0.5, -0.3, 0, 0.3, 0.5]
lambda_values = np.linspace(0, 5, 20)
fold_size = 5

# Load dataset
dirName = f'./datasets/regression/{dataset}'
test_data = pd.read_csv(f'{dirName}/test.csv')
target = data_target[dataset]
X_test = test_data.drop(target, axis=1)
y_test = test_data[target]
advice = np.array(data_advice[dataset].split(','), dtype=int)

# --- BASELINES: Scikit-GB and XGBoost ---
sgb_mse_folds = np.zeros(fold_size)
xgb_mse_folds = np.zeros(fold_size)

for fold in range(fold_size):
    train_data = pd.read_csv(f"{dirName}/train_{fold}.csv")
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]

    # SGB
    sgb = GradientBoostingRegressor(criterion='squared_error',
                                    n_estimators=30,
                                    max_depth=10,
                                    learning_rate=0.1,
                                    loss='ls',
                                    random_state=12)
    sgb.fit(X_train, y_train)
    sgb_mse_folds[fold] = mean_squared_error(y_test, sgb.predict(X_test))

    # XGBoost
    xgb_reg = xgb.XGBRegressor(n_estimators=30,
                               max_depth=10,
                               learning_rate=0.1,
                               objective='reg:squarederror',
                               verbosity=0,
                               random_state=12)
    xgb_reg.fit(X_train, y_train)
    xgb_mse_folds[fold] = mean_squared_error(y_test, xgb_reg.predict(X_test))

sgb_mse = np.mean(sgb_mse_folds)
xgb_mse = np.mean(xgb_mse_folds)

# --- SKiGB Grid Search and Plot ---
plt.figure(figsize=(8, 6))

for eps in epsilon_values:
    mse_list = []
    for lam in lambda_values:
        mse_folds = np.zeros(fold_size)
        for fold in range(fold_size):
            train_data = pd.read_csv(f"{dirName}/train_{fold}.csv")
            X_train = train_data.drop(target, axis=1)
            y_train = train_data[target]

            skigb = SKiGB(criterion='squared_error',
                          n_estimators=30,
                          max_depth=10,
                          learning_rate=0.1,
                          loss='ls',
                          random_state=12,
                          advice=advice,
                          lamda=lam,
                          epsilon=eps)
            skigb.fit(X_train, y_train)
            y_pred = skigb.kigb.predict(X_test)
            mse_folds[fold] = mean_squared_error(y_test, y_pred)

        mean_mse = np.mean(mse_folds)
        print(f"{-mean_mse}")
        mse_list.append(-mean_mse)  # negative MSE for upward trend
    plt.plot(lambda_values, mse_list, linestyle='--', label=f'ε: {eps:.1f}')

# Plot baselines
plt.axhline(y=-sgb_mse, color='green', linestyle='-', marker='o', label='SGB', linewidth=2)
plt.axhline(y=-xgb_mse, color='black', linestyle=':', marker='^', label='XGBoost', linewidth=2)

# Final touches
plt.title(dataset)
plt.xlabel("λ")
plt.ylabel("negative MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
