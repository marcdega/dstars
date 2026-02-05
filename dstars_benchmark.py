import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from dstars import DSTARS, aCC, aRMSE, aRRMSE
import time

# --- Configurações do Benchmark ---
N_SAMPLES = 500
N_FEATURES = 15
N_TARGETS = 5
NOISE = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS_RF = 20 # Reduzido para agilizar o benchmark
N_FOLDS_DSTARS = 3   # Reduzido para agilizar o benchmark

print("Gerando dados sintéticos para regressão multi-alvo...")
X, y = make_regression(n_samples=N_SAMPLES, n_features=N_FEATURES, n_targets=N_TARGETS, noise=NOISE, random_state=RANDOM_STATE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"Formato dos dados de treino: X={X_train.shape}, y={y_train.shape}")
print(f"Formato dos dados de teste: X={X_test.shape}, y={y_test.shape}")

results = []

# --- 1. DSTARS (DSTARST - Cross-validation) ---
print("\nExecutando DSTARS (DSTARST - Cross-validation)...")
start_time = time.time()
base_rf_estimator = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1)
dstars_model_dstarst = DSTARS(
    base_estimator=base_rf_estimator,
    method='DSTARST',
    n_folds_tracking=N_FOLDS_DSTARS,
    phi=0.4,
    epsilon=1e-4,
    random_state=RANDOM_STATE
)
dstars_model_dstarst.fit(X_train, y_train)
y_pred_dstarst = dstars_model_dstarst.predict(X_test)
end_time = time.time()

results.append({
    'Método': 'DSTARS (DSTARST)',
    'aCC': aCC(y_test, y_pred_dstarst),
    'aRMSE': aRMSE(y_test, y_pred_dstarst),
    'aRRMSE': aRRMSE(y_test, y_pred_dstarst),
    'Tempo (s)': end_time - start_time
})

# --- 2. DSTARS (DSTARS - Bootstrap) ---
print("\nExecutando DSTARS (DSTARS - Bootstrap)...")
start_time = time.time()
base_rf_estimator = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1)
dstars_model_dstars = DSTARS(
    base_estimator=base_rf_estimator,
    method='DSTARS',
    epsilon=1e-4,
    random_state=RANDOM_STATE
)
dstars_model_dstars.fit(X_train, y_train)
y_pred_dstars = dstars_model_dstars.predict(X_test)
end_time = time.time()

results.append({
    'Método': 'DSTARS (Bootstrap)',
    'aCC': aCC(y_test, y_pred_dstars),
    'aRMSE': aRMSE(y_test, y_pred_dstars),
    'aRRMSE': aRRMSE(y_test, y_pred_dstars),
    'Tempo (s)': end_time - start_time
})

# --- 3. MultiOutputRegressor (Single Target) ---
print("\nExecutando MultiOutputRegressor (Single Target)...")
start_time = time.time()
multi_output_rf = MultiOutputRegressor(
    estimator=RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1)
)
multi_output_rf.fit(X_train, y_train)
y_pred_multi_output = multi_output_rf.predict(X_test)
end_time = time.time()

results.append({
    'Método': 'MultiOutputRegressor (RF)',
    'aCC': aCC(y_test, y_pred_multi_output),
    'aRMSE': aRMSE(y_test, y_pred_multi_output),
    'aRRMSE': aRRMSE(y_test, y_pred_multi_output),
    'Tempo (s)': end_time - start_time
})

# --- 4. RegressorChain ---
print("\nExecutando RegressorChain...")
start_time = time.time()
chain_rf = RegressorChain(
    base_estimator=RandomForestRegressor(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1),
    order=None, # Ordem aleatória
    random_state=RANDOM_STATE
)
chain_rf.fit(X_train, y_train)
y_pred_chain = chain_rf.predict(X_test)
end_time = time.time()

results.append({
    'Método': 'RegressorChain (RF)',
    'aCC': aCC(y_test, y_pred_chain),
    'aRMSE': aRMSE(y_test, y_pred_chain),
    'aRRMSE': aRRMSE(y_test, y_pred_chain),
    'Tempo (s)': end_time - start_time
})

# --- Apresentar Resultados ---
results_df = pd.DataFrame(results)
print("\n--- Resultados do Benchmark ---")
print(results_df.round(4).to_markdown(index=False))

# Salvar resultados para análise posterior
results_df.to_csv("dstars_benchmark_results.csv", index=False)
print("\nResultados salvos em dstars_benchmark_results.csv")
