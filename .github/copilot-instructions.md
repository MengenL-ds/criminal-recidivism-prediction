# COMPAS Recidivism Classification Project

## Project Overview
This is a supervised machine learning classification project analyzing the COMPAS dataset to predict recidivism. The project is part of a Coursera portfolio and uses multiple ML models with grid search and SMOTE for class imbalance.

## Architecture & Data Flow

1. **Raw Data** → `data/unprocessed/compas-scores-two-years.csv` (ProPublica COMPAS dataset)
2. **EDA & Processing** → `notebooks/EDA.ipynb` (filtering, feature engineering, encoding)
3. **Processed Data** → `data/processed/compas_processed.parquet` (one-hot encoded, ready for modeling)
4. **Modeling** → `notebooks/modeling.ipynb` (12 models with GridSearchCV)
5. **Model Artifacts** → `models/*.pkl` (saved GridSearch objects for all models)

**Key insight**: Processed data is already one-hot encoded with `drop_first=True`, so modeling notebooks should load the parquet directly without re-encoding.

## Critical Data Processing Rules

The COMPAS dataset requires **specific filtering** based on ProPublica's methodology (implemented in `EDA.ipynb`):

```python
# Always apply these filters to raw data
df = df[df["days_b_screening_arrest"] <= 30]
df = df[df["days_b_screening_arrest"] >= -30]
df = df[df["is_recid"] != -1]
df = df[df["c_charge_degree"] != "O"]  # Remove ordinary traffic offenses
df['days_b_screening_arrest'] = df['days_b_screening_arrest'].clip(lower=0)
```

**Feature engineering pattern**: Jail duration is calculated as `jail_days = (c_jail_out - c_jail_in).dt.days`, then datetime columns are dropped.

## Model Training Conventions

### Pipeline Structure (Required)
All models must follow this 3-step pipeline pattern:

```python
Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),           # Handle class imbalance
    ('preprocess', ColumnTransformer(...)),       # Scale numerical features
    ('model', estimator)
])
```

### Features Requiring Scaling
```python
scale_feats = ["priors_count", "days_b_screening_arrest", "jail_days"]
```

### Class Imbalance Handling
- **SMOTE** is applied within the pipeline (random_state=42)
- Most models use `class_weight="balanced"` parameter
- XGBoost uses `scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()`
- CatBoost uses `class_weights=[1, (y_train == 0).sum() / (y_train == 1).sum()]`

### Model Ensemble
The stacking classifier uses:
```python
estimators = [
    ('svm_rbf', SVC(kernel='rbf', class_weight='balanced', probability=True)),
    ('logreg', LogisticRegression(class_weight='balanced')),
    ('knn', KNeighborsClassifier())
]
final_estimator = LogisticRegression()
```

## Model Evaluation Strategy

- **Train-test split**: 80/20 with `stratify=y, random_state=100`
- **Grid search CV**: 10-fold cross-validation, scoring metric is **recall** (prioritizing minority class detection)
- **Evaluation focus**: Recall for class 1 (recidivism) is the primary metric due to imbalanced dataset
- **Model persistence**: GridSearch objects are pickled to `models/{model_name}_gridsearch.pkl`

## Project-Specific Patterns

1. **Empty files are placeholders**: `src/EDA.py` and `README.md` exist but are empty—all work is in notebooks
2. **No requirements.txt**: Dependencies are evident from imports (sklearn, imblearn, lightgbm, xgboost, catboost, pandas, seaborn)
3. **Processed data directory**: `data/processed/` is empty in repo but populated during EDA execution
4. **CatBoost artifacts**: `notebooks/catboost_info/` contains training logs—ignore these files

## Common Commands

```bash
# Run notebooks (from project root)
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/modeling.ipynb

# View processed data structure (requires running EDA first)
python -c "import pandas as pd; print(pd.read_parquet('data/processed/compas_processed.parquet').info())"
```

## When Adding New Models

1. Add to the `models` dictionary with appropriate class_weight settings
2. Define hyperparameter grid in `params` dictionary using `model__` prefix
3. Ensure the model is wrapped in the SMOTE → Preprocessing → Model pipeline
4. Model will auto-save to `models/{model_name}_gridsearch.pkl` after GridSearch
5. Results will be appended to the `results` dictionary for comparison
