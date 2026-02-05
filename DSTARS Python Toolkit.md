# DSTARS Python Toolkit

Esta é uma reescrita em Python do toolkit **DSTARS** (Deep Structure for Tracking Asynchronous Regressor Stacking), originalmente desenvolvido em R por Saulo Martiello Mastelini.

## Funcionalidades

- **DSTARS**: Versão baseada em amostragem Bootstrap e Out-of-Bag (OOB) para determinar a estrutura de camadas.
- **DSTARST**: Versão baseada em Validação Cruzada (Cross-validation) com parâmetro de limiar `phi`.
- Compatível com a API do **scikit-learn** (`fit`, `predict`).
- Suporta qualquer regressor base do scikit-learn.
- Métricas de avaliação multi-alvo incluídas: `aCC`, `aRMSE`, `aRRMSE`.

## Instalação

Certifique-se de ter as seguintes dependências instaladas:

```bash
pip install numpy pandas scikit-learn
```

## Como usar

```python
from dstars import DSTARS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Carregue seus dados (X e y multi-alvo)
# X, y = ...

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Inicialize o modelo
# method='DSTARST' (padrão) ou 'DSTARS'
model = DSTARS(
    base_estimator=RandomForestRegressor(),
    method='DSTARST',
    n_folds_tracking=10,
    phi=0.4,
    epsilon=1e-4
)

# Treine o modelo
model.fit(X_train, y_train)

# Faça predições
y_pred = model.predict(X_test)

# Verifique as camadas de convergência encontradas para cada alvo
print(model.convergence_layers_)
```

## Métricas Incluídas

O toolkit fornece funções para calcular métricas comuns em Regressão Multi-alvo (MTR):

- `aCC(y_true, y_pred)`: Average Correlation Coefficient.
- `aRMSE(y_true, y_pred)`: Average Root Mean Squared Error.
- `aRRMSE(y_true, y_pred)`: Average Relative Root Mean Squared Error.

## Referência Original

Mastelini, S. M., Barbon Jr, S., & Santana, E. J. (2020). DSTARS: A multi-target deep structure for tracking asynchronous regressor stacking. *Applied Soft Computing*.
