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
│   ├── Random Forest_gridsearch.pkl
│   ├── xgboost_gridsearch. pkl
│   ├── catboost_gridsearch.pkl
│   └── ...  (12 models total)
├── src/
│   ├── EDA. py               # Utility functions for EDA
│   └── __init__.py
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis
- **Data quality checks**:  Missing values, duplicates, data types
- **Distribution analysis**: Skewness and kurtosis of numerical features
- **Demographic exploration**: Race, gender, and age distributions
- **Target variable analysis**: Recidivism rates across different groups
- **Feature relationships**: Correlation with target variable

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

### 4. Handling Class Imbalance
Addressed class imbalance through multiple strategies:
- **SMOTE (Synthetic Minority Over-sampling Technique)** for training data
- **Class weighting** for algorithms that support it
- **Custom weight parameters** for XGBoost and CatBoost
- Artificially increased imbalance (20% recidivists) to test robustness

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

## 📈 Key Findings

1. **Feature Importance**: `priors_count` consistently emerged as the strongest predictor of recidivism across all models, confirming that criminal history is a critical risk factor.

2. **Class Imbalance Impact**: Models using SMOTE and class weighting showed significantly better recall on the minority class (recidivists) compared to baseline approaches.

3. **Model Performance**:  Ensemble methods (Random Forest, XGBoost, CatBoost) generally outperformed simpler models, though at the cost of interpretability.

4. **Demographic Patterns**: The dataset shows racial disparities (51% African-American, 34% Caucasian) that are important to consider when deploying such models in real-world contexts.

5. **Age Effects**:  Younger defendants (<25) with prior convictions showed higher recidivism rates, validating the `young_with_priors` engineered feature.

## 🚀 Getting Started

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

## 📝 Usage

### Loading Processed Data
```python
import pandas as pd

# Load the preprocessed dataset
df = pd.read_parquet('data/processed/compas_processed.parquet', engine='fastparquet')
```

### Loading Trained Models
```python
import pickle

# Load a trained model
with open('models/xgboost_gridsearch. pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X_test)
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
- **Human Oversight**: Predictive models should augment, not replace, human judgment in criminal justice contexts.

## References

- ProPublica COMPAS Analysis: [Machine Bias](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
- Original COMPAS Dataset: [ProPublica GitHub](https://github.com/propublica/compas-analysis)
- SMOTE Paper:  Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"

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

*Built with care for demonstrating practical machine learning workflows in sensitive domains.*
