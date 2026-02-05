# Relatório Comparativo: DSTARS vs. Outros Métodos de Regressão Multi-Alvo em Python

## Introdução

A Regressão Multi-Alvo (MTR) é uma tarefa de aprendizado de máquina onde um único modelo é treinado para prever múltiplas variáveis de saída correlacionadas. O toolkit DSTARS (Deep Structure for Tracking Asynchronous Regressor Stacking), originalmente desenvolvido em R, propõe uma abordagem de empilhamento em camadas para explorar as dependências entre os alvos e melhorar o desempenho preditivo. Este relatório apresenta uma comparação do desempenho da implementação em Python do DSTARS com outros métodos MTR comuns no ecossistema Python: MultiOutputRegressor (abordagem de alvo único independente) e RegressorChain.

## Metodologia

Para a comparação, foram gerados dados sintéticos de regressão multi-alvo utilizando a função `make_regression` do scikit-learn, com 500 amostras, 15 features de entrada e 5 variáveis alvo, com adição de ruído. Os dados foram divididos em conjuntos de treino e teste na proporção de 80/20, respectivamente.

Os seguintes modelos foram avaliados:

1.  **DSTARS (DSTARST)**: A versão de validação cruzada do DSTARS, utilizando `RandomForestRegressor` como estimador base (`n_estimators=20`). Configurado com `n_folds_tracking=3`, `phi=0.4` e `epsilon=1e-4`.
2.  **DSTARS (Bootstrap)**: A versão de bootstrap do DSTARS, utilizando `RandomForestRegressor` como estimador base (`n_estimators=20`). Configurado com `epsilon=1e-4`.
3.  **MultiOutputRegressor (RF)**: Um `MultiOutputRegressor` do scikit-learn, que treina um regressor independente para cada alvo. Utilizou `RandomForestRegressor` (`n_estimators=20`) como estimador base.
4.  **RegressorChain (RF)**: Um `RegressorChain` do scikit-learn, que modela as dependências entre os alvos treinando regressores em uma sequência, onde as previsões dos alvos anteriores são usadas como features para os alvos subsequentes. Utilizou `RandomForestRegressor` (`n_estimators=20`) como estimador base e ordem aleatória dos alvos.

As métricas de avaliação utilizadas foram:

-   **aCC (Average Correlation Coefficient)**: Média dos coeficientes de correlação entre os valores reais e previstos para cada alvo. Valores mais altos indicam melhor desempenho.
-   **aRMSE (Average Root Mean Squared Error)**: Média das Raízes do Erro Quadrático Médio para cada alvo. Valores mais baixos indicam melhor desempenho.
-   **aRRMSE (Average Relative Root Mean Squared Error)**: Média das Raízes do Erro Quadrático Médio Relativo para cada alvo. Valores mais baixos indicam melhor desempenho.
-   **Tempo (s)**: Tempo de execução do treinamento e predição para cada modelo.

## Resultados

A tabela abaixo resume o desempenho de cada método no conjunto de teste:

| Método | aCC | aRMSE | aRRMSE | Tempo (s) |
|:--------------------------|-------:|--------:|---------:|------------:|
| DSTARS (DSTARST) | 0.8717 | 79.9847 | 0.4999 | 10.2079 |
| DSTARS (Bootstrap) | 0.8704 | 80.3316 | 0.5014 | 3.8883 |
| MultiOutputRegressor (RF) | 0.8646 | 85.8662 | 0.5356 | 0.2756 |
| RegressorChain (RF) | 0.8169 | 97.1056 | 0.6107 | 0.3377 |

## Discussão

Os resultados demonstram que ambas as versões do DSTARS (DSTARST e Bootstrap) superaram os métodos `MultiOutputRegressor` e `RegressorChain` em termos de precisão preditiva (aCC, aRMSE e aRRMSE). O `DSTARS (DSTARST)` obteve o melhor desempenho geral, com o maior aCC e os menores valores de aRMSE e aRRMSE, indicando uma melhor capacidade de capturar as relações complexas entre os alvos.

É importante notar que o custo computacional do DSTARS é significativamente maior em comparação com o `MultiOutputRegressor` e o `RegressorChain`. Isso se deve à sua natureza iterativa e ao empilhamento de múltiplos regressores em camadas, especialmente na versão DSTARST que realiza validação cruzada interna. No entanto, para aplicações onde a precisão é crítica e o tempo de treinamento pode ser tolerado, o DSTARS se mostra uma alternativa promissora.

O `MultiOutputRegressor` apresentou um desempenho razoável, mas inferior ao DSTARS, o que é esperado, pois ele não explora as dependências entre os alvos. O `RegressorChain`, embora tente modelar essas dependências, teve o pior desempenho neste benchmark, possivelmente devido à ordem aleatória dos alvos ou à complexidade dos dados sintéticos.

## Conclusão

A implementação em Python do toolkit DSTARS demonstrou ser eficaz na melhoria do desempenho da regressão multi-alvo, superando os métodos tradicionais de `MultiOutputRegressor` e `RegressorChain` em termos de precisão. Embora o custo computacional seja maior, os benefícios em termos de acurácia podem justificar seu uso em cenários onde a modelagem precisa das interdependências entre os alvos é crucial. A escolha entre DSTARS (DSTARST) e DSTARS (Bootstrap) dependerá do equilíbrio desejado entre precisão e tempo de execução, sendo o DSTARST geralmente mais preciso, mas mais lento.

## Referências

[1] Mastelini, S. M., Barbon Jr, S., & Santana, E. J. (2020). DSTARS: A multi-target deep structure for tracking asynchronous regressor stacking. *Applied Soft Computing*.
