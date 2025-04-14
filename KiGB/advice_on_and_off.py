from core.lgbm.lkigb import LKiGB as KiGB
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Load data
train_data = pd.read_csv('datasets/classification/adult/train_0.csv')
test_data = pd.read_csv('datasets/classification/adult/test.csv')
X_train = train_data.drop('income', axis=1)
Y_train = train_data['income']
X_test = test_data.drop('income', axis=1)
Y_test = test_data['income']

# Define monotonic advice
real_advice = np.array([-1, -1, 0, +1, 0, +1], dtype=int)
no_advice = np.zeros_like(real_advice)

# Train with real advice
model_with_advice = KiGB(lamda=1, epsilon=0.1, advice=real_advice, objective='binary', trees=30)
model_with_advice.fit(X_train, Y_train)
preds_with_advice = model_with_advice.predict(X_test)
acc_with = accuracy_score(Y_test, preds_with_advice)

# Train with no advice
model_no_advice = KiGB(lamda=1, epsilon=0.1, advice=no_advice, objective='binary', trees=30)
model_no_advice.fit(X_train, Y_train)
preds_no_advice = model_no_advice.predict(X_test)
acc_withincome = accuracy_score(Y_test, preds_no_advice)

# incomeput results
print(f"Accuracy with advice:    {acc_with:.4f}")
print(f"Accuracy without advice: {acc_withincome:.4f}")
