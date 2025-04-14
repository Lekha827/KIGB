import lightgbm as lgb

baseline = lgb.LGBMClassifier(objective='binary', n_estimators=30)
baseline.fit(X_train, Y_train)
Y_pred_baseline = baseline.predict(X_test)

from sklearn.metrics import classification_report, accuracy_score

print("🔹 KiGB Metrics:")
print(classification_report(Y_test, Y_pred))

print("🔹 LightGBM Baseline Metrics:")
print(classification_report(Y_test, Y_pred_baseline))

print("🔹 Accuracy Comparison:")
print("KiGB:", accuracy_score(Y_test, Y_pred))
print("LightGBM:", accuracy_score(Y_test, Y_pred_baseline))