import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import lightgbm as lgb

try:
    from core.lgbm.lkigb import LKiGB
    from experiments.regression.setting import *
except ImportError:
    sys.path.append('./')
    from core.lgbm.lkigb import LKiGB
    from experiments.regression.setting import *

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

# --- BASELINES: LGBM (no constraints) and LMC (with constraints) ---
lgbm_score = np.zeros(fold_size)
lmc_score = np.zeros(fold_size)

for fold in range(fold_size):
    train_data = pd.read_csv(f"{dirName}/train_{fold}.csv")
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]

    # LGBM baseline (no constraints)
    lgbm_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'l2',
        'learning_rate': 0.1,
        'max_depth': 14,
        'verbosity': -1
    }
    lgb_train = lgb.Dataset(X_train, y_train, free_raw_data=False)
    lgbm = lgb.train(lgbm_params, lgb_train, num_boost_round=30)
    lgbm_score[fold] = mean_squared_error(y_test, lgbm.predict(X_test))

    # LMC (LightGBM with monotonic constraints)
    lmc_params = lgbm_params.copy()
    lmc_params['monotone_constraints'] = data_advice[dataset]
    lmc = lgb.train(lmc_params, lgb_train, num_boost_round=30)
    lmc_score[fold] = mean_squared_error(y_test, lmc.predict(X_test))

lgbm_mse = np.mean(lgbm_score)
lmc_mse = np.mean(lmc_score)

# --- SKiGB-style Grid Search (LKiGB) ---
plt.figure(figsize=(10, 7))

for eps in epsilon_values:
    mse_list = []
    for lam in lambda_values:
        mse_folds = np.zeros(fold_size)
        for fold in range(fold_size):
            train_data = pd.read_csv(f"{dirName}/train_{fold}.csv")
            X_train = train_data.drop(target, axis=1)
            y_train = train_data[target]

            lkigb = LKiGB(lamda=lam,
                          epsilon=eps,
                          max_depth=14,
                          advice=advice,
                          objective='regression',
                          trees=30)
            lkigb.fit(X_train, y_train)
            y_pred = lkigb.predict(X_test)
            mse_folds[fold] = mean_squared_error(y_test, y_pred)

        mean_mse = np.mean(mse_folds)
        mse_list.append(-mean_mse)  # negative MSE for upward trend

    plt.plot(lambda_values, mse_list, linestyle='--', label=f'ε: {eps:.1f}')

# Plot LGBM and LMC baselines
plt.axhline(y=-lgbm_mse, color='green', linestyle='-', marker='o', label='LGBM', linewidth=2)
plt.axhline(y=-lmc_mse, color='black', linestyle=':', marker='^', label='LMC', linewidth=2)

# Final touches
plt.title(dataset)
plt.xlabel("λ")
plt.ylabel("negative MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
