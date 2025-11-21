import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
from itertools import combinations
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action='ignore')


def prepare_data_leave_one_out(df_list2, df_out_index):
    train_df = pd.concat([df for i, df in enumerate(df_list2) if i != df_out_index])
    test_df = df_list2[df_out_index]
    return train_df, test_df


def regression_and_RMSE(train_df, test_df, features,
                        target_col='Interpolated Values_Enregy expendiures'):
    """
    features: string (one column) or list of strings (multiple columns)
    """
    # Make sure features is a list
    if isinstance(features, str):
        features = [features]

    X_train = train_df[features].values
    y_train = train_df[target_col].values

    X_test = test_df[features].values
    y_test = test_df[target_col].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # -------------------------

    # Fit regression model
    reg = LinearRegression()
    reg.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = reg.predict(X_test_scaled)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return rmse, y_pred, y_test

def evaluate_per_activity(train_df, test_df, features,
                          target_col='Interpolated Values_Enregy expendiures',
                          activity_col='Activity'):
  

    # Train + predict using the universal function
    fold_rmse, y_pred, y_test = regression_and_RMSE(
        train_df, test_df, features, target_col=target_col
    )

    activities = test_df[activity_col].values

    activity_rmse = {}
    activity_counts = {}

    for act in np.unique(activities):
        mask = (activities == act)
        n = np.sum(mask)
        if n == 0:
            continue

        rmse = mean_squared_error(y_test[mask], y_pred[mask], squared=False)
        activity_rmse[act] = rmse
        activity_counts[act] = n

    return activity_rmse, activity_counts, fold_rmse, y_pred, y_test