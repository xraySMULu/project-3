import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib  # For saving and loading models


def train_or_load_model(data_path, model_path='interest_rate_model.pkl', train=True):
    """
    Initializes, trains, and evaluates a model or loads a pretrained model.

    Args:
        data_path (str): Path to the CSV file containing interest rate data.
        model_path (str, optional): Path to save/load the model. Defaults to 'interest_rate_model.pkl'.
        train (bool, optional): Whether to train a new model or load an existing one. Defaults to True.

    Returns:
        tuple: A tuple containing the trained/loaded model and the RMSE on the training data.
    """

    # Load the data
    Interest_rates = pd.read_csv(data_path)
    Interest_rates.dropna(inplace=True)
    Interest_rate = Interest_rates.rename(
        columns={'date': 'DATE', ' value': 'Rate'})
    Interest_rate['DATE'] = pd.to_datetime(Interest_rate['DATE'])

    # Filter data
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2024, 12, 31)
    filtered_rates = Interest_rate[(Interest_rate['DATE'] >= start_date) & (
        Interest_rate['DATE'] <= end_date)]
    filtered_rates['year_month'] = filtered_rates['DATE'].dt.strftime('%Y-%m')
    grouped_rates_ym = filtered_rates.groupby(
        ['year_month'])['Rate'].mean().reset_index()
    Final_rates = grouped_rates_ym

    if train:
        # Prepare data for modeling
        X = Final_rates.index.values.reshape(-1, 1)
        y = Final_rates['Rate']

        # Time Series Split
        tscv = TimeSeriesSplit(n_splits=5)

        # Model and hyperparameter tuning
        model = LinearRegression()
        param_grid = {'fit_intercept': [True, False]}
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Save the model
        joblib.dump(best_model, model_path)

    else:
        # Load the pretrained model
        best_model = joblib.load(model_path)

    # Make predictions and evaluate
    Final_rates['Predicted_Rate'] = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(
        Final_rates['Rate'], Final_rates['Predicted_Rate']))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    return best_model, rmse

# Example usage:
# To train a new model:
# model, rmse = train_or_load_model('fed-rates.csv', train=True)

# To load a pretrained model:
# model, rmse = train_or_load_model('fed-rates.csv', train=False)
