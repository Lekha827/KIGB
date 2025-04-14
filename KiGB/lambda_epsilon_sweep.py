import pandas as pd
import numpy as np
from core.lgbm.lkigb import LKiGB as KiGB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('datasets/classification/car/train_0.csv')
test_data = pd.read_csv('datasets/classification/car/test.csv')
X_train = train_data.drop('class', axis=1)
Y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
Y_test = test_data['class']

# Monotonic advice
advice = np.array([-1, -1, 0, +1, 0, +1], dtype=int)

# Define parameter ranges
lamda_vals = [0.1, 1, 10]
epsilon_vals = [0.01, 0.1, 1]

results = []

# Grid search over lamda and epsilon
for lamda in lamda_vals:
    for epsilon in epsilon_vals:
        kigb = KiGB(lamda=lamda, epsilon=epsilon, advice=advice, objective='binary', trees=30)
        kigb.fit(X_train, Y_train)
        Y_pred = kigb.predict(X_test)
        acc = accuracy_score(Y_test, Y_pred)
        results.append({'lamda': lamda, 'epsilon': epsilon, 'accuracy': acc})
        print(f"λ={lamda}, ε={epsilon} -> Accuracy: {acc:.4f}")

# Plot heatmap of results
import seaborn as sns

results_df = pd.DataFrame(results)
pivot = results_df.pivot(index='lamda', columns='epsilon', values='accuracy')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".4f", cmap="YlGnBu")
plt.title("KiGB Accuracy vs λ and ε")
plt.xlabel("Epsilon (ε)")
plt.ylabel("Lambda (λ)")
plt.tight_layout()
plt.show()
