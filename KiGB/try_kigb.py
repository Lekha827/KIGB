import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from core.lgbm.lkigb import LKiGB as KiGB

# Step 1: Load Dataset
train_data = pd.read_csv('datasets/classification/car/train_0.csv')
test_data = pd.read_csv('datasets/classification/car/test.csv')

X_train = train_data.drop('class', axis=1)
Y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
Y_test = test_data['class']

# Step 2: Define Monotonic Advice
advice = np.array([-1, -1, 0, +1, 0, +1], dtype=int)

# Step 3: Train KiGB Model
kigb = KiGB(lamda=1, epsilon=0.1, advice=advice, objective='binary', trees=30)
kigb.fit(X_train, Y_train)
Y_pred_kigb = kigb.predict(X_test)

# Step 4: Train LightGBM Model (Baseline)
lgbm = lgb.LGBMClassifier(n_estimators=30, objective='binary', random_state=42)
lgbm.fit(X_train, Y_train)
Y_pred_lgbm = lgbm.predict(X_test)

# Step 5: Evaluate Both Models
def evaluate_model(name, y_true, y_pred):
    print(f"\n--- {name} ---")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model("KiGB", Y_test, Y_pred_kigb)
evaluate_model("LightGBM", Y_test, Y_pred_lgbm)

# Step 6: Plot Confusion Matrices
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

plot_conf_matrix(Y_test, Y_pred_kigb, "KiGB")
plot_conf_matrix(Y_test, Y_pred_lgbm, "LightGBM")


# import lightgbm as lgb
# import matplotlib.pyplot as plt

# print(type(kigb.kigb))
# print(kigb.kigb.num_trees())
# # lgb.plot_tree(kigb.kigb, tree_index=22, figsize=(20, 20))
# lgb.plot_tree(kigb.kigb, tree_index=0, figsize=(20, 10))

# plt.show()


import lightgbm as lgb

baseline = lgb.LGBMClassifier(objective='binary', n_estimators=30)
baseline.fit(X_train, Y_train)
Y_pred_baseline = baseline.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

print("🔹 KiGB Metrics:")
print(classification_report(Y_test, Y_pred_kigb))

print("🔹 LightGBM Baseline Metrics:")
print(classification_report(Y_test, Y_pred_baseline))

print("🔹 Accuracy Comparison:")
print("KiGB:", accuracy_score(Y_test, Y_pred_kigb))
print("LightGBM:", accuracy_score(Y_test, Y_pred_baseline))