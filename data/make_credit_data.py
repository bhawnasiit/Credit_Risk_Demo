import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

rng = np.random.default_rng(42)
N = 5000  # number of applicants

def truncated_normal(mean, sd, low, high, size):
    vals = rng.normal(mean, sd, size)
    return np.clip(vals, low, high)

credit_data = pd.DataFrame({
    "age": truncated_normal(35, 10, 18, 70, N).astype(int),
    "income": truncated_normal(60000, 20000, 20000, 150000, N).astype(int),
    "loan_amount": truncated_normal(15000, 8000, 1000, 50000, N).astype(int),
    "credit_score": truncated_normal(650, 70, 300, 850, N).astype(int),
    "employment_length": rng.integers(0, 20, N),
    "delinquencies": rng.integers(0, 5, N),
})

linear = (
    5
    - 0.005 * credit_data["credit_score"]
    - 0.00003 * credit_data["income"]
    + 0.0001 * credit_data["loan_amount"]
    + 0.25 * credit_data["delinquencies"]
)
p_default = 1 / (1 + np.exp(-linear))
credit_data["default"] = rng.binomial(1, p_default)

Path("data").mkdir(parents=True, exist_ok=True)
credit_data.to_csv("data/credit_data.csv", index=False)
print("✅ Wrote data/credit_data.csv with", len(credit_data), "rows")


def perform_eda(df, save_plots=True):
    """
    Perform comprehensive Exploratory Data Analysis on the credit dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        save_plots (bool): Whether to save plots to files
    """
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - CREDIT RISK DATASET")
    print("=" * 60)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Basic Dataset Information
    print("\n1. DATASET OVERVIEW")
    print("-" * 30)
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data types:\n{df.dtypes}")
    
    # 2. Missing Data Analysis
    print("\n2. MISSING DATA ANALYSIS")
    print("-" * 30)
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    })
    print(missing_df)
    
    # 3. Basic Statistical Summary
    print("\n3. STATISTICAL SUMMARY")
    print("-" * 30)
    print(df.describe())
    
    # 4. Target Variable Analysis
    print("\n4. TARGET VARIABLE ANALYSIS")
    print("-" * 30)
    if 'default' in df.columns:
        default_counts = df['default'].value_counts()
        default_percent = df['default'].value_counts(normalize=True) * 100
        print(f"Default distribution:")
        print(f"  No Default (0): {default_counts[0]} ({default_percent[0]:.2f}%)")
        print(f"  Default (1): {default_counts[1]} ({default_percent[1]:.2f}%)")
        print(f"Class imbalance ratio: {default_counts[0]/default_counts[1]:.2f}:1")
    
    # 5. Feature Distributions
    print("\n5. FEATURE DISTRIBUTIONS")
    print("-" * 30)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Skewness: {stats.skew(df[col]):.3f}")
        print(f"  Kurtosis: {stats.kurtosis(df[col]):.3f}")
    
    # 6. Correlation Analysis
    print("\n6. CORRELATION ANALYSIS")
    print("-" * 30)
    correlation_matrix = df[numeric_cols].corr()
    print("Correlation Matrix:")
    print(correlation_matrix.round(3))
    
    # Create visualizations
    if save_plots:
        create_eda_plots(df)
    
    return {
        'shape': df.shape,
        'missing_data': missing_df,
        'correlation_matrix': correlation_matrix,
        'target_distribution': default_counts if 'default' in df.columns else None
    }

def create_eda_plots(df):
    """Create and save EDA visualization plots."""
    
    # Create output directory for plots
    Path("eda_plots").mkdir(exist_ok=True)
    
    # 1. Feature Distribution Plots
    print("\nCreating distribution plots...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('eda_plots/feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box Plots for Outlier Detection
    print("Creating box plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(numeric_cols):
        if i < len(axes):
            axes[i].boxplot(df[col])
            axes[i].set_title(f'Box Plot of {col}')
            axes[i].set_ylabel(col)
    
    for i in range(len(numeric_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('eda_plots/box_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Correlation Heatmap
    print("Creating correlation heatmap...")
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('eda_plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Target Variable Analysis
    if 'default' in df.columns:
        print("Creating target variable analysis...")
        
        # Default rate by different features
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Default rate by age groups
        df['age_group'] = pd.cut(df['age'], bins=5, labels=['18-28', '29-38', '39-48', '49-58', '59-70'])
        default_by_age = df.groupby('age_group', observed=True)['default'].mean()
        axes[0,0].bar(default_by_age.index, default_by_age.values)
        axes[0,0].set_title('Default Rate by Age Group')
        axes[0,0].set_ylabel('Default Rate')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Default rate by income groups
        df['income_group'] = pd.cut(df['income'], bins=5)
        default_by_income = df.groupby('income_group', observed=True)['default'].mean()
        axes[0,1].bar(range(len(default_by_income)), default_by_income.values)
        axes[0,1].set_title('Default Rate by Income Group')
        axes[0,1].set_ylabel('Default Rate')
        axes[0,1].set_xticks(range(len(default_by_income)))
        axes[0,1].set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                                  for interval in default_by_income.index], rotation=45)
        
        # Default rate by credit score groups
        df['credit_score_group'] = pd.cut(df['credit_score'], bins=5)
        default_by_credit = df.groupby('credit_score_group', observed=True)['default'].mean()
        axes[1,0].bar(range(len(default_by_credit)), default_by_credit.values)
        axes[1,0].set_title('Default Rate by Credit Score Group')
        axes[1,0].set_ylabel('Default Rate')
        axes[1,0].set_xticks(range(len(default_by_credit)))
        axes[1,0].set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                                  for interval in default_by_credit.index], rotation=45)
        
        # Default rate by loan amount groups
        df['loan_amount_group'] = pd.cut(df['loan_amount'], bins=5)
        default_by_loan = df.groupby('loan_amount_group', observed=True)['default'].mean()
        axes[1,1].bar(range(len(default_by_loan)), default_by_loan.values)
        axes[1,1].set_title('Default Rate by Loan Amount Group')
        axes[1,1].set_ylabel('Default Rate')
        axes[1,1].set_xticks(range(len(default_by_loan)))
        axes[1,1].set_xticklabels([f'{int(interval.left)}-{int(interval.right)}' 
                                 for interval in default_by_loan.index], rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_plots/target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Clean up temporary columns
        df.drop(['age_group', 'income_group', 'credit_score_group', 'loan_amount_group'], 
                axis=1, inplace=True)
    
    # 5. Pairwise Scatter Plots
    print("Creating pairwise scatter plots...")
    if len(numeric_cols) > 1:
        # Select a subset of features for pairwise plots to avoid overcrowding
        key_features = ['age', 'income', 'credit_score', 'loan_amount', 'default']
        available_features = [col for col in key_features if col in df.columns]
        
        if len(available_features) >= 2:
            sns.pairplot(df[available_features], hue='default' if 'default' in available_features else None)
            plt.savefig('eda_plots/pairwise_scatter.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ EDA plots saved to 'eda_plots/' directory")

# Main execution
if __name__ == "__main__":
    # Generate the dataset
    rng = np.random.default_rng(42)
    N = 5000  # number of applicants

    def truncated_normal(mean, sd, low, high, size):
        vals = rng.normal(mean, sd, size)
        return np.clip(vals, low, high)

    credit_data = pd.DataFrame({
        "age": truncated_normal(35, 10, 18, 70, N).astype(int),
        "income": truncated_normal(60000, 20000, 20000, 150000, N).astype(int),
        "loan_amount": truncated_normal(15000, 8000, 1000, 50000, N).astype(int),
        "credit_score": truncated_normal(650, 70, 300, 850, N).astype(int),
        "employment_length": rng.integers(0, 20, N),
        "delinquencies": rng.integers(0, 5, N),
    })

    linear = (
        5
        - 0.005 * credit_data["credit_score"]
        - 0.00003 * credit_data["income"]
        + 0.0001 * credit_data["loan_amount"]
        + 0.25 * credit_data["delinquencies"]
    )
    p_default = 1 / (1 + np.exp(-linear))
    credit_data["default"] = rng.binomial(1, p_default)

    # Save the dataset
    Path("data").mkdir(parents=True, exist_ok=True)
    credit_data.to_csv("data/credit_data.csv", index=False)
    print("✅ Wrote data/credit_data.csv with", len(credit_data), "rows")
    
    # Perform EDA
    eda_results = perform_eda(credit_data, save_plots=True)
    
    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)
