inflation-forecasting-morocco/
│
├── README.md                          # Description principale
├── requirements.txt                   # Dépendances Python
├── environment.R                      # Packages R nécessaires
│
├── data
│
├── notebooks/
│   ├── 01_exploratory_analysis.R     # Exploration et visualisation
│   ├── 02_stationarity_tests.R       # Tests ADF, KPSS
│   ├── 03_arima_modeling.R           # Modélisation ARIMA
│   └── 04_ml_models.ipynb            # Modèles ML (Random Forest, Ridge)
│
├── src/
│   ├── data_preparation.py           # Fonctions de preprocessing
│   ├── feature_engineering.py        # Création des features (lags, rolling)
│   ├── arima_utils.R                 # Fonctions utilitaires ARIMA
│   └── ml_models.py                  # Classes pour RF et Ridge
│
├── results/
│   ├── figures/                      # Graphiques générés
│   │   ├── acf_pacf_plot.png
│   │   ├── forecast_arima.png
│   │   └── forecast_ridge.png
│   └── metrics/
│       └── model_comparison.csv      # Tableau comparatif des performances
│
└── docs/

    └── methodology.md                # Documentation méthodologique
