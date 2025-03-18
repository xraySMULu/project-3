import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import csv
import os

def plot_top_correlation_heatmap(df, target_column="SalePrice", top_n=15):
    """
    Plots a correlation heatmap for the top N features most correlated with the target column.
    
    Parameters:
    df (pd.DataFrame): The input dataframe with encoded features.
    target_column (str): The column for which correlation is computed.
    top_n (int): The number of top correlated features to include in the heatmap.
    
    Returns:
    None
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Identify top N most important features related to target_column
    top_features = corr_matrix[target_column].abs().sort_values(ascending=False).head(top_n + 1)[1:].index
    print(f"Top {top_n} Features Most Correlated with {target_column}:")
    print(top_features.tolist())
    
    # Plot heatmap for top N features
    plt.figure(figsize=(12, 10))
    top_corr_matrix = corr_matrix.loc[top_features, top_features]
    sns.heatmap(top_corr_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)
    plt.title(f"Correlation Heatmap of Top {top_n} Features Related to {target_column}")
    plt.show()

    return top_features

def train_linear_regression(df, top_features, target_column="SalePrice"):
    """
    Trains and evaluates a Linear Regression model using the top correlated features.

    Parameters:
    df (pd.DataFrame): The input dataframe with encoded features.
    top_features (list): List of feature names to use for training.
    target_column (str): The target variable.

    Returns:
    dict: Performance metrics including MAE, MSE, and R2 score.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    # Select features and target
    X = df[top_features]
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    # Evaluate performance
    metrics = {
        "MAE": mean_absolute_error(y_test, lr_preds),
        "MSE": mean_squared_error(y_test, lr_preds),
        "R2 Score": r2_score(y_test, lr_preds),
    }

    print("Linear Regression Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\n")



def train_random_forest(df, top_features, target_column="SalePrice", n_estimators=100, random_state=42):
    """
    Trains and evaluates a Random Forest Regressor model using the top correlated features.

    Parameters:
    df (pd.DataFrame): The input dataframe with encoded features.
    top_features (list): List of feature names to use for training.
    target_column (str): The target variable.
    n_estimators (int): The number of trees in the forest.
    random_state (int): Random seed for reproducibility.

    Returns:
    dict: Performance metrics including MAE, MSE, and R2 score.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    # Select features and target
    X = df[top_features]
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # Evaluate performance
    metrics = {
        "MAE": mean_absolute_error(y_test, rf_preds),
        "MSE": mean_squared_error(y_test, rf_preds),
        "R2 Score": r2_score(y_test, rf_preds),
    }

    print("Random Forest Performance:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("\n")

def optimize_random_forest(df, top_features, target_column="SalePrice", param_grid=None, search_type="grid", output_csv="../Resources/data/rf_tuning_results.csv"):
    """
    Optimizes a Random Forest model using Grid Search or Randomized Search, logs all results to a CSV, and prints the best parameters.

    Parameters:
    df (pd.DataFrame): The input dataframe with encoded features.
    top_features (list): List of feature names to use for training.
    target_column (str): The target variable.
    param_grid (dict): Hyperparameter search space.
    search_type (str): "grid" for GridSearchCV or "random" for RandomizedSearchCV.
    output_csv (str): Path to save tuning results.

    Returns:
    pd.DataFrame: DataFrame containing all parameter combinations and performance metrics.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")

    # Select features and target
    X = df[top_features]
    y = df[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter search space if not provided
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }

    # Choose search method
    if search_type == "grid":
        search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="r2", n_jobs=-1, return_train_score=True)
    elif search_type == "random":
        search = RandomizedSearchCV(RandomForestRegressor(random_state=42), param_distributions=param_grid, n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42, return_train_score=True)
    else:
        raise ValueError("Invalid search_type. Choose 'grid' or 'random'.")

    # Fit the model with hyperparameter tuning
    search.fit(X_train, y_train)

    # Extract all results from hyperparameter tuning
    results = pd.DataFrame(search.cv_results_)

    # Select relevant columns for logging
    results = results[[
        "param_n_estimators", "param_max_depth", "param_min_samples_split", "param_min_samples_leaf",
        "mean_test_score", "mean_train_score", "std_test_score"
    ]]
    
    results.rename(columns={
        "param_n_estimators": "n_estimators",
        "param_max_depth": "max_depth",
        "param_min_samples_split": "min_samples_split",
        "param_min_samples_leaf": "min_samples_leaf",
        "mean_test_score": "Mean R2 Score (Test)",
        "mean_train_score": "Mean R2 Score (Train)",
        "std_test_score": "R2 Score Std Dev"
    }, inplace=True)

    # Save all parameter results to CSV
    results.to_csv(output_csv, mode="a", index=False, header=not os.path.exists(output_csv))

    # Print the best hyperparameters
    best_params = search.best_params_
    best_model = search.best_estimator_

    # Make predictions with the best model
    predictions = best_model.predict(X_test)

    # Evaluate performance of the best model
    best_metrics = {
        "Best Params": best_params,
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "R2 Score": r2_score(y_test, predictions),
    }

    print(f"Optimized Random Forest Performance ({search_type.capitalize()} Search):")
    print(best_metrics)

    print("\nAll tuning results saved to:", output_csv)
    

