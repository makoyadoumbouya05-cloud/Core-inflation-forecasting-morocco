inflation-forecasting-morocco/
│
├── README.md                          # Description principale
├── requirements.txt                   # Dépendances Python
├── environment.R                      # Packages R nécessaires
│
├── data
│  
├── notebooks/
│   ├── 01\_exploratory\_analysis.R     # Exploration et visualisation
│   ├── 02\_stationarity\_tests.R       # Tests ADF, KPSS
│   ├── 03\_arima\_modeling.R           # Modélisation ARIMA
│   └── 04\_ml\_models.ipynb            # Modèles ML (Random Forest, Ridge)
│
├── src/
│   ├── data\_preparation.py           # Fonctions de preprocessing
│   ├── feature\_engineering.py        # Création des features (lags, rolling)
│   ├── arima\_utils.R                 # Fonctions utilitaires ARIMA
│   └── ml\_models.py                  # Classes pour RF et Ridge
│
├── results/
│   ├── figures/                      # Graphiques générés
│   │   ├── acf\_pacf\_plot.png
│   │   ├── forecast\_arima.png
│   │   └── forecast\_ridge.png
│   └── metrics/
│       └── model\_comparison.csv      # Tableau comparatif des performances
│
└── docs/
└── methodology.md                # Documentation méthodologique

