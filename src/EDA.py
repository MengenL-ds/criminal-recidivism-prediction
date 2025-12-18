import matplotlib.pyplot as plt
import seaborn as sns

def plot_feat_histogram(df, column_name, bins=10):
    """Plot histogram of a numerical feature and print skewness and kurtosis"""
    plt.figure(figsize=(5, 3))
    sns.histplot(df[column_name], bins=bins, kde=False)
    plt.title(f'Histogram of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    print(f' skewness: {df[column_name].skew():.2f}')
    print(f' kurtosis: {df[column_name].kurtosis():.2f}')

def skew_kurtosis(df, column_name):
    """Calculate and print skewness and kurtosis of a numerical feature"""
    print(f'Feature: {column_name}')
    print(f' skewness: {df[column_name].skew():.2f}')
    print(f' kurtosis: {df[column_name].kurtosis():.2f}')
    if df[column_name].skew() > 0.5 and df[column_name].kurtosis() > 3:
        print(" The feature is positively skewed with high kurtosis.")
    else:
        print(" The feature does NOT exhibit strong positive skewness and high kurtosis.")