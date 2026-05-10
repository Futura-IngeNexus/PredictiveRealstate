# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Modelo de Machine Learning para predecir el precio de alquiler de departamentos en la Ciudad Autónoma de Buenos Aires (CABA). El trabajo completo vive en un único notebook:

```
Proyecto_PredictivoInmobiliario.ipynb
```

## Datos

El dataset proviene del repositorio público [MVernetti/Alquileres_CABA](https://github.com/MVernetti/Alquileres_CABA). Para ejecutar el notebook en Google Colab, clonarlo primero:

```bash
!git clone https://github.com/MVernetti/Alquileres_CABA.git
```

El CSV principal es `2alquileres_CABA.csv` (149 859 filas, 30 columnas, separador `|`, encoding `latin-1`).

**Variable objetivo:** `alq_dolar` — precio de alquiler en USD.

## Ejecución

El notebook está diseñado para Google Colab. Ejecutar las celdas en orden; cada sección depende de la anterior:

| Sección | Contenido |
|---------|----------|
| 1 | Importación de librerías |
| 2 | Carga de datos |
| 3 | Exploración (EDA) |
| 4 | Preprocesamiento |
| 5 | Reducción de dimensionalidad (PCA) |
| 6 | Modelado (6 modelos) |
| 7 | Selección del modelo |
| 8 | Conclusiones |
| 9 | Función de predicción |

## Arquitectura del pipeline

```
CSV raw
  └─► Renombrado de columnas
  └─► Eliminación de duplicados e IDs repetidos
  └─► Limpieza de nulos y outliers por variable
        ├── antiguedad: IQR + imputación por mediana de barrio
        ├── ambientes:  límites dominio + mediana global
        └── superficie: regresión lineal sobre ambientes
  └─► Encoding categórico
        └── barrio → categoria_barrio (0=bajo, 1=medio, 2=alto)
  └─► PCA sobre amenidades binarias
        [gimnasio, lavadero, pileta, balcon, vigilancia] → PC1, PC2, PC3
  └─► MinMaxScaler (TODAS las variables, incluida la target)
  └─► Modelos
        ├── Ridge (alpha=0.01)
        ├── Polinómica grado 3
        ├── SVR kernel linear
        ├── Decision Tree (max_depth=20, max_features='log2')
        ├── Random Forest (n_estimators=500, max_depth=10)
        └── XGBoost (HalvingGridSearchCV)
```

## Decisiones de diseño importantes

- **Target escalada con MinMaxScaler**: `alq_dolar` y `alq_peso` se escalan junto con las features. Las métricas (MSE, R²) se calculan sobre valores normalizados; para interpretar en USD se debe usar `scaler.inverse_transform`.
- **PCA sobre amenidades**: las variables `gimnasio`, `lavadero`, `pileta`, `balcon`, `vigilancia` se comprimen en 3 componentes principales (≈80% varianza explicada) antes del modelado lineal y SVR.
- **`categoria_barrio`**: los barrios de CABA se agrupan en 3 categorías de precio; el mapeo está hardcodeado en la celda de la Sección 4.1.1.
- **Variables descartadas**: `cochera` (64% nulos), `lat`, `lng`, `fecha_pub_aviso`, `exp`, `sup_tot`, `solarium`, `frente`, `contrafrente`, `ascensor`.
- **Variables duplicadas eliminadas**: `alq_arg` ≈ `alq_peso`; `alq_usd` ≈ `alq_dolar` (se conserva la columna con menos nulos de cada par).

## Dependencias principales

```
pandas, numpy, matplotlib, seaborn, statsmodels
scikit-learn >= 1.1  (HalvingGridSearchCV, PolynomialFeatures, PCA)
xgboost
hyperopt
scipy
```

## Variables globales clave (estado del notebook)

| Variable | Definida en | Descripción |
|----------|------------|-------------|
| `df` | Sección 2 | DataFrame principal (se modifica in-place a lo largo del notebook) |
| `scaler` | Sección 4.3 | `MinMaxScaler` ajustado sobre el df preprocesado |
| `r2_test_poly` | Sección 6.2 | R² prueba — regresión polinómica |
| `r2_test_arbol` | Sección 6.4 | R² prueba — árbol de decisión |
| `r2_test_RF` | Sección 6.5 | R² prueba — Random Forest |
| `r2_test_xgb` | Sección 6.6 | R² prueba — XGBoost |
| `best_xgb` | Sección 6.6 | Modelo XGBoost final entrenado |
| `rf_model` | Sección 6.5 | Modelo Random Forest final entrenado |

## Función de predicción (Sección 9)

```python
predecir_alquiler(modelo, scaler_obj, superficie, ambientes,
                  gimnasio=0, pileta=0, vigilancia=0,
                  categoria_barrio=1, lavadero=0)
# → retorna precio estimado en USD (float)
```

Requiere el `scaler` ajustado y el modelo entrenado (`best_xgb` o `rf_model`). Aplica la normalización internamente usando los índices del scaler: `superficie=2`, `ambientes=3`, `categoria_barrio=6`, `alq_dolar=0`.
