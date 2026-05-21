# Analýza rizika bankrotu podnikov pomocou dátovej analytiky

## Popis práce

Táto práca  je zameraná na predikciu pravdepodobnosti bankrotu podnikov. Súčasťou je aj interaktívna aplikácia vytvorená v prostredí Streamlit, ktorá umožňuje vykonávať **What-if analýzu** finančných ukazovateľov.
Používateľ môže meniť vybrané finančné ukazovatele a sledovať, ako sa zmení pravdepodobnosť bankrotu podniku.
V projekte boli testované a porovnávané viaceré modely strojového učenia a techniky riešenia nevyvážených dát.

Práca bola vytvorená v jazyku Python v prostredí Jupyter Notebook. Súbory je možné otvoriť v Jupyter Notebook alebo vo Visual Studio Code.

## Použité knižnice

### Spracovanie dát
- pandas
- numpy

### Strojové učenie
- scikit-learn
- xgboost
- lightgbm
- imbalanced-learn

### Vizualizácia
- matplotlib
- seaborn

### Webová aplikácia
- streamlit

### Ukladanie modelov
- joblib



## Štruktúra projektu

### Jupyter notebooky

| Súbor | Popis |
|---|---|
| `XGBoost.ipynb` | Model XGBoost a jeho optimalizácia |
| `RF.ipynb` | Model Random Forest |
| `LightGBM.ipynb` | Model LightGBM |
| `KNN.ipynb` | Model K-Nearest Neighbors |
| `Adaboost.ipynb` | Model AdaBoost |
| `analyza_dat.ipynb` | Analýza a vizualizácia dát |
| `model_pre_aplikaciu.ipynb` | Príprava finálneho modelu pre aplikáciu |


### Súbory aplikácie

| Súbor | Popis |
|---|---|
| `app.py` | Streamlit aplikácia pre What-if analýzu |
| `bankruptcy_model.pkl` | Uložený natrénovaný model |
| `best_threshold.pkl` | Optimálny rozhodovací prah |
| `feature_names.pkl` | Zoznam použitých premenných |


### Dátové súbory

| Súbor | Popis |
|---|---|
| `data.csv` | Pôvodný dataset |
| `X_test.csv` | Testovacie vstupné dáta |
| `y_test.csv` | Testovacie cieľové hodnoty |


## Funkcionalita aplikácie

Aplikácia umožňuje:

- vybrať podnik z množiny,
- meniť finančné ukazovatele podniku,
- simulovať rôzne finančné scenáre,
- sledovať zmenu pravdepodobnosti bankrotu,
- nastaviť vlastný rozhodovací threshold,
- porovnať pôvodný a simulovaný stav,
- zobraziť vizualizáciu výsledkov.


## Vyhodnotenie modelov

Modely boli hodnotené pomocou metrík:

- Classification Report
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve



## Spustenie aplikácie

Na spustenie aplikácie použite:

```bash
streamlit run app.py
```

Yana Pavlyk

