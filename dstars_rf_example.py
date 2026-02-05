import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from dstars import DSTARS, aCC, aRMSE, aRRMSE

print("Gerando dados sintéticos para regressão multi-alvo...")
# Gerar dados sintéticos para regressão multi-alvo
# n_samples: número de amostras
# n_features: número de features de entrada
# n_targets: número de variáveis alvo a serem previstas
# noise: desvio padrão do ruído gaussiano adicionado aos alvos
# random_state: semente para reprodutibilidade
X, y = make_regression(n_samples=500, n_features=15, n_targets=5, noise=0.5, random_state=42)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Formato dos dados de treino: X={X_train.shape}, y={y_train.shape}")
print(f"Formato dos dados de teste: X={X_test.shape}, y={y_test.shape}")

print("\nInicializando e treinando o modelo DSTARS com RandomForestRegressor (método DSTARST - Cross-validation)...")
# Inicializar o RandomForestRegressor como estimador base
base_rf_estimator = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)

# Inicializar o modelo DSTARS com o estimador base
# method=\'DSTARST\' (padrão) utiliza validação cruzada interna para determinar as camadas
# n_folds_tracking: número de folds para a validação cruzada interna
# phi: limiar para seleção de camadas (0.4 é um bom ponto de partida)
# epsilon: mínimo decréscimo de erro esperado para adicionar uma nova camada
dstars_model_dstarst = DSTARS(
    base_estimator=base_rf_estimator,
    method='DSTARST',
    n_folds_tracking=5, # Reduzido para agilizar o exemplo
    phi=0.4,
    epsilon=1e-4,
    random_state=42
)

# Treinar o modelo DSTARS
dstars_model_dstarst.fit(X_train, y_train)

print("Treinamento DSTARST concluído. Realizando predições...")
# Fazer predições no conjunto de teste
y_pred_dstarst = dstars_model_dstarst.predict(X_test)

print("\nAvaliando o desempenho do DSTARS (DSTARST)...")
# Avaliar o desempenho usando as métricas multi-alvo
acc_dstarst = aCC(y_test, y_pred_dstarst)
rmse_dstarst = aRMSE(y_test, y_pred_dstarst)
rrmse_dstarst = aRRMSE(y_test, y_pred_dstarst)

print(f"Resultados DSTARS (DSTARST) com RandomForestRegressor:")
print(f"  Average Correlation Coefficient (aCC): {acc_dstarst:.4f}")
print(f"  Average Root Mean Squared Error (aRMSE): {rmse_dstarst:.4f}")
print(f"  Average Relative Root Mean Squared Error (aRRMSE): {rrmse_dstarst:.4f}")
print(f"  Camadas de Convergência por Alvo: {dstars_model_dstarst.convergence_layers_}")

print("\nInicializando e treinando o modelo DSTARS com RandomForestRegressor (método DSTARS - Bootstrap)...")
# Inicializar o modelo DSTARS com o estimador base (método DSTARS - Bootstrap)
# method=\'DSTARS\' utiliza bootstrap e OOB para determinar as camadas
dstars_model_dstars = DSTARS(
    base_estimator=base_rf_estimator,
    method='DSTARS',
    epsilon=1e-4,
    random_state=42
)

# Treinar o modelo DSTARS
dstars_model_dstars.fit(X_train, y_train)

print("Treinamento DSTARS (Bootstrap) concluído. Realizando predições...")
# Fazer predições no conjunto de teste
y_pred_dstars = dstars_model_dstars.predict(X_test)

print("\nAvaliando o desempenho do DSTARS (Bootstrap)...")
# Avaliar o desempenho usando as métricas multi-alvo
acc_dstars = aCC(y_test, y_pred_dstars)
rmse_dstars = aRMSE(y_test, y_pred_dstars)
rrmse_dstars = aRRMSE(y_test, y_pred_dstars)

print(f"Resultados DSTARS (Bootstrap) com RandomForestRegressor:")
print(f"  Average Correlation Coefficient (aCC): {acc_dstars:.4f}")
print(f"  Average Root Mean Squared Error (aRMSE): {rmse_dstars:.4f}")
print(f"  Average Relative Root Mean Squared Error (aRRMSE): {rrmse_dstars:.4f}")
print(f"  Camadas de Convergência por Alvo: {dstars_model_dstars.convergence_layers_}")

print("\nComparando com um RandomForestRegressor padrão (sem DSTARS)...")
# Comparar com um RandomForestRegressor padrão (sem a lógica DSTARS)
standard_rf_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
standard_rf_model.fit(X_train, y_train)
y_pred_standard_rf = standard_rf_model.predict(X_test)

acc_standard_rf = aCC(y_test, y_pred_standard_rf)
rmse_standard_rf = aRMSE(y_test, y_pred_standard_rf)
rrmse_standard_rf = aRRMSE(y_test, y_pred_standard_rf)

print(f"Resultados RandomForestRegressor Padrão:")
print(f"  Average Correlation Coefficient (aCC): {acc_standard_rf:.4f}")
print(f"  Average Root Mean Squared Error (aRMSE): {rmse_standard_rf:.4f}")
print(f"  Average Relative Root Mean Squared Error (aRRMSE): {rrmse_standard_rf:.4f}")

print("\nExemplo concluído.")
