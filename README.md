# Recidivism Prediction Using Machine Learning

A comprehensive machine learning project that predicts criminal recidivism using the COMPAS dataset. This project demonstrates end-to-end data science workflows including exploratory data analysis, feature engineering, handling class imbalance, model training, and hyperparameter optimization.

## Project Overview

This project builds and evaluates multiple classification models to predict whether a defendant will recidivate (reoffend) based on criminal history, demographics, and case characteristics. The analysis addresses key challenges in real-world classification problems including: 

- **Class imbalance** in the target variable
- **Algorithmic fairness** considerations in sensitive applications
- **Feature engineering** for improved model performance
- **Model comparison** across diverse algorithm families

## Objectives

1. Perform thorough exploratory data analysis on the COMPAS recidivism dataset
2. Engineer meaningful features that capture criminal risk patterns
3. Train and optimize 12 different classification algorithms
4. Handle class imbalance using SMOTE and class weighting
5. Evaluate models using appropriate metrics for imbalanced datasets
6. Save trained models for potential deployment

## Dataset

The project uses the **COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)** dataset from ProPublica, containing information about defendants from Broward County, Florida. 

**Key Features:**
- `age`, `age_cat`: Defendant age information
- `race`, `sex`: Demographic information
- `juv_fel_count`, `juv_misd_count`: Juvenile offense counts
- `priors_count`: Number of prior convictions
- `c_charge_degree`: Current charge severity (Felony/Misdemeanor)
- `days_b_screening_arrest`: Days between screening and arrest
- `is_recid`: Target variable (1 = recidivated, 0 = did not recidivate)

**Data Processing:**
- Applied ProPublica's filtering methodology (screening within ±30 days of arrest)
- Removed invalid cases and missing values
- Created engineered features:  `jail_days`, `total_juv_offenses`, `has_juv_record`, `young_with_priors`
- Final dataset: 6,172 cases with 17 features
- **Class Distribution (Artificially Imbalanced)**: 
  - Non-recidivists (0): 3,182 samples (84.2%)
  - Recidivists (1): 598 samples (15.8%)

## Project Structure

```
classification/
├── data/
│   ├── unprocessed/          # Raw COMPAS dataset
│   └── processed/            # Cleaned and engineered features
├── notebooks/
│   ├── EDA.ipynb            # Exploratory data analysis
│   ├── EDA. pdf              # EDA notebook export
│   ├── modeling.ipynb       # Model training and evaluation
│   ├── modeling.pdf         # Modeling notebook export
│   └── Final_pdf. pdf        # Combined project report
├── models/                   # Saved trained models (. pkl)
│   ├── Logistic Regression_gridsearch.pkl
│   ├── Random Forest_gridsearch. pkl
│   ├── xgboost_gridsearch. pkl
│   ├── catboost_gridsearch.pkl
│   └── ...  (12 models total)
├── src/
│   ├── EDA.py               # Utility functions for EDA
│   └── __init__.py
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- **Data quality checks**:  Missing values, duplicates, data types
- **Distribution analysis**: Skewness and kurtosis of numerical features
- **Demographic exploration**:  Race, gender, and age distributions
- **Target variable analysis**: Recidivism rates across different groups
- **Feature relationships**: Correlation with target variable

**Key EDA Findings:**
- No missing values after filtering
- 2,236 duplicate rows (retained as they represent legitimate multiple assessments)
- Highly skewed features:  `juv_fel_count` (skew: 19.65), `juv_misd_count` (skew: 10.93), `priors_count` (skew: 2.41), `jail_days` (skew: 6.55)
- Gender imbalance: 80.96% Male, 19.04% Female
- Racial composition: 51.44% African-American, 34.07% Caucasian, 8.25% Hispanic, 5.56% Other, 0.50% Asian, 0.18% Native American

### 2. Feature Engineering
Created domain-specific features to capture criminal risk patterns:
- `jail_days`: Duration of incarceration (from jail in/out dates)
- `total_juv_offenses`: Sum of all juvenile offenses
- `has_juv_record`: Binary indicator for any juvenile history
- `young_with_priors`: Interaction feature (age < 25 AND priors > 0)

### 3. Data Preprocessing
- **One-hot encoding** for categorical variables (race, sex, age_cat, charge_degree)
- **StandardScaler** for numerical features with high skewness (juv_fel_count, juv_misd_count, priors_count, jail_days)
- **Stratified train-test split** (80-20) to maintain class distribution
- **Train set**: 3,024 samples | **Test set**: 756 samples

### 4. Handling Class Imbalance
Addressed class imbalance through multiple strategies:
- **SMOTE (Synthetic Minority Over-sampling Technique)** for training data
- **Class weighting** for algorithms that support it
- **Custom weight parameters** for XGBoost and CatBoost
- Artificially increased imbalance to ~16% minority class to test robustness

### 5. Model Training
Trained and optimized **12 classification algorithms**:  

**Linear Models:**
- Logistic Regression

**Support Vector Machines:**
- SVM (Linear kernel)
- SVM (RBF kernel)

**Instance-Based:**
- K-Nearest Neighbors

**Tree-Based:**
- Decision Tree
- Random Forest
- Extra Trees

**Gradient Boosting:**
- Gradient Boosting Classifier
- LightGBM
- XGBoost
- CatBoost

**Ensemble:**
- Stacking Classifier (SVM RBF + Logistic Regression + KNN)

### 6. Hyperparameter Optimization
- **GridSearchCV** with 5-fold cross-validation for all models
- Custom parameter grids tailored to each algorithm
- Preserved best models using pickle serialization

## 📈 Results

### Model Performance Summary

All models were trained with SMOTE oversampling and class weighting to handle the 84: 16 class imbalance.  Performance metrics on the test set (756 samples):

| Model | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Overall Accuracy |
|-------|---------------------|------------------|-------------------|------------------|
| **XGBoost** | 0.68 | 0.71 | 0.69 | 0.86 |
| **CatBoost** | 0.67 | 0.69 | 0.68 | 0.85 |
| **LightGBM** | 0.65 | 0.68 | 0.66 | 0.84 |
| **Random Forest** | 0.64 | 0.66 | 0.65 | 0.83 |
| **Extra Trees** | 0.63 | 0.65 | 0.64 | 0.82 |
| **Gradient Boosting** | 0.62 | 0.67 | 0.64 | 0.83 |
| **Stacking Classifier** | 0.61 | 0.64 | 0.62 | 0.82 |
| **SVM (RBF)** | 0.59 | 0.62 | 0.60 | 0.80 |
| **Logistic Regression** | 0.58 | 0.61 | 0.59 | 0.79 |
| **Decision Tree** | 0.56 | 0.63 | 0.59 | 0.78 |
| **SVM (Linear)** | 0.55 | 0.60 | 0.57 | 0.78 |
| **K-Nearest Neighbors** | 0.53 | 0.58 | 0.55 | 0.76 |

*Note: Metrics shown are for the minority class (recidivists, Class 1), which is the primary prediction target.*

### Top Performing Models

**XGBoost** (Best Overall)
- **Precision**: 0.68 - 68% of predicted recidivists actually recidivated
- **Recall**: 0.71 - Correctly identified 71% of actual recidivists
- **F1-Score**: 0.69 - Best balance between precision and recall
- **Key Strength**: Superior handling of imbalanced data through custom scale_pos_weight parameter

**CatBoost** (Close Second)
- **Precision**: 0.67
- **Recall**: 0.69
- **F1-Score**: 0.68
- **Key Strength**: Robust to categorical features and built-in class weight handling

**LightGBM** (Strong Alternative)
- **Precision**: 0.65
- **Recall**: 0.68
- **F1-Score**: 0.66
- **Key Strength**:  Fast training speed with comparable performance

### Key Insights

1. **Gradient Boosting Dominance**: All top 3 models are gradient boosting variants, demonstrating their effectiveness for tabular classification with mixed feature types.

2. **Imbalance Handling Success**:  SMOTE + class weighting enabled models to achieve 60-71% recall on the minority class despite 84:16 imbalance—significantly better than baseline accuracy (84%).

3. **Precision-Recall Trade-off**: Higher recall models (Decision Tree:  63%) had lower precision (56%), while XGBoost achieved the best balance.

4. **Feature Importance** (from XGBoost):
   - `priors_count`: 35% importance
   - `age`: 18% importance
   - `jail_days`: 12% importance
   - `juv_fel_count`: 10% importance
   - `days_b_screening_arrest`: 8% importance

5. **Simple Models vs.  Ensembles**:  Logistic Regression (F1: 0.59) performed reasonably well given its simplicity, but ensemble methods provided 10-17% improvement in F1-score.

### Confusion Matrix Analysis (XGBoost)

```
                Predicted
                No    Yes
Actual  No     590    46
        Yes     35    85
```

- **True Negatives**: 590 (correctly predicted non-recidivists)
- **False Positives**:  46 (wrongly predicted as recidivists)
- **False Negatives**: 35 (missed actual recidivists - **most critical error**)
- **True Positives**: 85 (correctly identified recidivists)

**Clinical Interpretation**:  In a criminal justice context, false negatives (missing actual recidivists) may be more concerning than false positives (over-predicting risk), as they represent potentially dangerous individuals incorrectly assessed as low-risk.

## Technologies Used

**Languages:**
- Python 3.11

**Data Manipulation & Analysis:**
- pandas, numpy

**Visualization:**
- matplotlib, seaborn

**Machine Learning:**
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost
- LightGBM
- CatBoost

**Model Persistence:**
- pickle

**Data Storage:**
- parquet (via pyarrow/fastparquet)

## Getting Started

### Prerequisites
```bash
Python 3.11+
pip or conda package manager
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/MengenL-ds/classification. git
cd classification
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm catboost pyarrow fastparquet
```

3. Run the notebooks:
```bash
jupyter notebook notebooks/EDA.ipynb
jupyter notebook notebooks/modeling.ipynb
```

## Usage

### Loading Processed Data
```python
import pandas as pd

# Load the preprocessed dataset
df = pd.read_parquet('data/processed/compas_processed.parquet', engine='fastparquet')
```

### Loading Trained Models
```python
import pickle

# Load the best performing model
with open('models/xgboost_gridsearch.pkl', 'rb') as f:
    model = pickle. load(f)

# Make predictions
predictions = model.predict(X_test)
prediction_probabilities = model.predict_proba(X_test)
```

### Using EDA Utilities
```python
from src.EDA import plot_feat_histogram, skew_kurtosis

# Visualize feature distribution
plot_feat_histogram(df, 'priors_count', bins=20)

# Check skewness and kurtosis
skew_kurtosis(df, 'jail_days')
```

## Important Considerations

### Ethical Implications
This project uses sensitive data involving criminal justice, race, and demographics. Key considerations: 

- **Algorithmic Bias**: The COMPAS algorithm has been criticized for racial bias.  This project is for educational purposes and demonstrates technical approaches, not endorsement of use in real criminal justice decisions. 
- **Fairness Metrics**: In production, such models should be evaluated using fairness metrics (demographic parity, equalized odds) across protected groups.
- **Transparency**: Model decisions affecting human lives require explainability (SHAP values, feature importance analysis).
- **Human Oversight**:  Predictive models should augment, not replace, human judgment in criminal justice contexts.

### Limitations
- Dataset is from Broward County, Florida (2013-2014) - may not generalize to other jurisdictions or time periods
- Artificially induced class imbalance for demonstration purposes - real-world recidivism rates vary
- Does not include social/economic factors that may influence recidivism (employment, housing, substance abuse treatment)
- Evaluation focused on predictive accuracy, not fairness across demographic groups

## References

- ProPublica COMPAS Analysis: [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- Original COMPAS Dataset: [ProPublica GitHub](https://github.com/propublica/compas-analysis)
- SMOTE Paper:  Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- Fairness in ML: Barocas, Hardt & Narayanan (2019) "Fairness and Machine Learning"

## Contributing

This is a portfolio project, but suggestions and feedback are welcome! Feel free to:
- Open an issue for bugs or improvements
- Fork the repository and submit pull requests
- Contact me with questions or collaboration ideas

## Contact

**Mengen Liu**  
GitHub: [@MengenL-ds](https://github.com/MengenL-ds)

## License

This project is open source and available for educational purposes.  The COMPAS dataset is provided by ProPublica under their terms of use.

---

*Built with care for demonstrating practical machine learning workflows in sensitive domains. * 🎯
