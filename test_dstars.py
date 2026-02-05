import numpy as np
import pandas as pd
from dstars import DSTARS, aCC, aRMSE, aRRMSE
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate synthetic multi-target data
X, y = make_regression(n_samples=200, n_features=10, n_targets=3, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Iniciando teste do DSTARS...")

# Initialize and fit DSTARS (DSTARST)
print("\nTestando DSTARST (Cross-validation)...")
model = DSTARS(n_folds_tracking=3, method='DSTARST', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"aCC: {aCC(y_test, y_pred):.4f}, aRMSE: {aRMSE(y_test, y_pred):.4f}, Camadas: {model.convergence_layers_}")

# Initialize and fit DSTARS (Bootstrap)
print("\nTestando DSTARS (Bootstrap)...")
model_bs = DSTARS(method='DSTARS', random_state=42)
model_bs.fit(X_train, y_train)
y_pred_bs = model_bs.predict(X_test)
print(f"aCC: {aCC(y_test, y_pred_bs):.4f}, aRMSE: {aRMSE(y_test, y_pred_bs):.4f}, Camadas: {model_bs.convergence_layers_}")

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = aCC(y_test, y_pred)
rmse = aRMSE(y_test, y_pred)
rrmse = aRRMSE(y_test, y_pred)

print(f"Resultados DSTARS:")
print(f"aCC: {acc:.4f}")
print(f"aRMSE: {rmse:.4f}")
print(f"aRRMSE: {rrmse:.4f}")
print(f"Camadas de convergência: {model.convergence_layers_}")

# Compare with standard Multi-target Random Forest
print("\nComparando com Random Forest padrão...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

acc_rf = aCC(y_test, y_pred_rf)
rmse_rf = aRMSE(y_test, y_pred_rf)
rrmse_rf = aRRMSE(y_test, y_pred_rf)

print(f"Resultados RF:")
print(f"aCC: {acc_rf:.4f}")
print(f"aRMSE: {rmse_rf:.4f}")
print(f"aRRMSE: {rrmse_rf:.4f}")
