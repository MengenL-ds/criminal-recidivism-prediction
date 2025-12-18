# Criminal Recidivism Prediction

A machine learning project focused on predicting criminal recidivism using various classification algorithms. This project analyzes offender data to predict the likelihood of reoffending, which can assist in criminal justice decision-making processes.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to predict whether an offender will recidivate (reoffend) using machine learning classification algorithms. The prediction model considers various features including demographics, criminal history, and other relevant factors to assess recidivism risk.

## Dataset

The dataset includes offender information with the following key features:
- **Demographics**: Age, gender, race
- **Criminal History**: Prior offenses, offense types
- **Risk Factors**: Various behavioral and social indicators
- **Target Variable**: Recidivism status (binary classification)

## Models Implemented

The following classification algorithms have been implemented and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Support Vector Machine (SVM)** - Linear and RBF kernels
5. **K-Nearest Neighbors (KNN)**
6. **Naive Bayes**
7. **Gradient Boosting Classifier**
8. **XGBoost**
9. **CatBoost**
10. **LightGBM**

## Results

### Model Performance Overview

The models were evaluated using multiple metrics including accuracy, precision, recall, F1-score, and ROC-AUC. Here are the comprehensive results:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **SVM (RBF)** | **0.8916** | **0.8862** | **0.8985** | **0.8923** | **0.8915** |
| **Gradient Boosting** | **0.8898** | **0.8831** | **0.8984** | **0.8907** | **0.8897** |
| **K-Nearest Neighbors** | **0.8862** | **0.8782** | **0.8964** | **0.8872** | **0.8861** |
| Random Forest | 0.8844 | 0.8754 | 0.8959 | 0.8855 | 0.8843 |
| LightGBM | 0.8826 | 0.8725 | 0.8953 | 0.8837 | 0.8824 |
| Logistic Regression | 0.8808 | 0.8697 | 0.8949 | 0.8821 | 0.8806 |
| SVM (Linear) | 0.8790 | 0.8669 | 0.8944 | 0.8805 | 0.8788 |
| Decision Tree | 0.8664 | 0.8508 | 0.8864 | 0.8682 | 0.8661 |
| Naive Bayes | 0.8646 | 0.8479 | 0.8859 | 0.8665 | 0.8643 |
| XGBoost | 0.8466 | 0.8193 | 0.8924 | 0.8543 | 0.8456 |
| CatBoost | 0.8430 | 0.8142 | 0.8919 | 0.8513 | 0.8419 |

### Top Performing Models

#### 1. **SVM with RBF Kernel** (Best Overall)
- **Accuracy**: 89.16%
- **Precision**: 88.62%
- **Recall**: 89.85%
- **F1-Score**: 89.23%
- **Strengths**: Excellent balance across all metrics, robust decision boundaries
- **Use Case**: Optimal for balanced prediction needs

#### 2. **Gradient Boosting Classifier** (Close Second)
- **Accuracy**: 88.98%
- **Precision**: 88.31%
- **Recall**: 89.84%
- **F1-Score**: 89.07%
- **Strengths**: Strong ensemble performance, handles complex patterns well
- **Use Case**: Reliable for production deployment

#### 3. **K-Nearest Neighbors** (Third Best)
- **Accuracy**: 88.62%
- **Precision**: 87.82%
- **Recall**: 89.64%
- **F1-Score**: 88.72%
- **Strengths**: Non-parametric approach, good for local patterns
- **Use Case**: Effective when similar cases matter

### Detailed Analysis

#### Precision-Recall Tradeoff

The results reveal an important pattern in model performance:

**High Precision Models** (SVM RBF, Gradient Boosting, KNN):
- Lower false positive rates
- More reliable when predicting recidivism
- Better suited for interventions with resource constraints
- Minimizes unnecessary interventions

**High Recall Models** (CatBoost, XGBoost):
- Catch more potential recidivists (fewer missed cases)
- Higher false positive rates
- Better suited when missing a true recidivist has high cost
- More conservative approach to public safety

#### Understanding CatBoost and XGBoost Performance

While **CatBoost** and **XGBoost** show lower overall accuracy (84.30% and 84.66%), they achieve the **highest recall rates** (89.19% and 89.24%):

**Why the Performance Pattern?**
- These models are optimized to minimize false negatives
- They predict recidivism more liberally, capturing more true positives
- Trade-off: Lower precision due to more false positives
- The models may be using default hyperparameters that favor recall

**When to Use Them:**
- When the cost of missing a recidivist is very high
- Early warning systems where you want maximum coverage
- Initial screening that will be followed by additional assessment

### Key Insights

1. **Balanced Performance**: SVM RBF achieves the best balance across all metrics, making it the most reliable all-around model.

2. **Ensemble Strength**: Traditional ensemble methods (Gradient Boosting, Random Forest) perform consistently well, validating their effectiveness for this task.

3. **Model Selection Matters**: The choice between models should depend on the specific use case:
   - **For balanced decisions**: Use SVM RBF or Gradient Boosting
   - **For maximum coverage**: Consider XGBoost or CatBoost
   - **For interpretability**: Logistic Regression or Decision Tree perform reasonably well

4. **Recidivism Prediction Challenges**: All models achieve 84%+ accuracy, suggesting the features contain predictive power, but the task remains challenging with inherent uncertainties.

5. **Marginal Differences**: The top 6 models are within 5% accuracy of each other, suggesting that feature engineering and data quality may be as important as model selection.

### Real-World Application Considerations

In the **criminal justice domain**, model selection has significant implications:

- **False Positives**: Incorrectly predicting recidivism could lead to harsher sentencing or denied parole for individuals who would not reoffend
- **False Negatives**: Missing true recidivists could pose public safety risks
- **Ethical Considerations**: Models must be regularly audited for bias and fairness across demographic groups
- **Human Oversight**: These predictions should inform, not replace, human judgment in decision-making
- **Transparency**: Stakeholders need to understand model limitations and confidence levels

**Recommendation**: Deploy **SVM RBF** or **Gradient Boosting** as primary models for their balanced performance, while maintaining **XGBoost/CatBoost** as sensitivity analysis tools to identify high-risk cases that might be missed by more conservative models.

## Installation

```bash
# Clone the repository
git clone https://github.com/MengenL-ds/classification.git
cd classification

# Install required packages
pip install -r requirements.txt
```

## Project Structure

```
classification/
│
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for analysis
├── src/               # Source code
│   ├── models/        # Model implementations
│   ├── preprocessing/ # Data preprocessing scripts
│   └── evaluation/    # Model evaluation utilities
├── results/           # Model results and visualizations
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This project is for educational and research purposes. When applying recidivism prediction models in real-world scenarios, careful consideration must be given to ethical implications, bias mitigation, and the appropriate use of predictive analytics in criminal justice systems.
