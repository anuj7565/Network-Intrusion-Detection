import joblib

model = joblib.load('rf_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

importances = model.feature_importances_
ranked = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)

for name, score in ranked:
    print(f"{name}: {score:.4f}")
print(len(feature_columns), len(importances))