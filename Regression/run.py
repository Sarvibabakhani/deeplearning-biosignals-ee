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


from prepare_train_evaluate_Regression import prepare_data_leave_one_out, regression_and_RMSE, evaluate_per_activity

def run_LOSO_cv(df_list2, features,
                target_col='Interpolated Values_Enregy expendiures'):

    rmse_values = []
    last_y_pred, last_y_test = None, None

    for j in range(len(df_list2)):
        train_df, test_df = prepare_data_leave_one_out(df_list2, j)
        rmse, y_pred, y_test = regression_and_RMSE(train_df, test_df, features, target_col)
        rmse_values.append(rmse)
        last_y_pred, last_y_test = y_pred, y_test

        print(f"Subject {j+1}: RMSE = {rmse}")

    mean_rmse = np.mean(rmse_values)
    # var_rmse = np.var(rmse_values)

    print("." * 50)
    print(features, f"\n\nMean RMSE: {mean_rmse}")
    print("-" * 50)

    return 



def run_LOSO_per_activity(df_list2, features,
                          target_col='Interpolated Values_Enregy expendiures',
                          activity_col='Activity Code'):
    total_activity_rmse = defaultdict(float)   # sum(rmse * count)
    total_activity_counts = defaultdict(int)   # total count
    fold_weighted_rmses = []

    for j in range(len(df_list2)):
        train_df, test_df = prepare_data_leave_one_out(df_list2, j)

        activity_rmse, activity_counts, fold_rmse, y_pred, y_test = evaluate_per_activity(
            train_df, test_df, features,
            target_col=target_col,
            activity_col=activity_col
        )

        # accumulate weighted sums per activity
        for act in activity_rmse:
            rmse = activity_rmse[act]
            n = activity_counts[act]
            total_activity_rmse[act] += rmse * n
            total_activity_counts[act] += n

        # weighted average RMSE for this subject
        weighted_sum = sum(activity_rmse[act] * activity_counts[act] for act in activity_rmse)
        total_samples = sum(activity_counts.values())
        weighted_avg_rmse = weighted_sum / total_samples
        fold_weighted_rmses.append(weighted_avg_rmse)

        # print(f"\nSubject left out: {j+1}")
        # for act in sorted(activity_rmse):
        #     print(f"  Activity {act}: RMSE = {activity_rmse[act]:.4f}, n = {activity_counts[act]}")
        # print(f"  Weighted Average RMSE: {weighted_avg_rmse:.4f}")

    # Final summary
    print("\n" + "=" * 60)
    
    print(features)

    per_activity_rmse = {}
    for act in sorted(total_activity_rmse):
        count = total_activity_counts[act]
        if count > 0:
            avg_rmse_act = total_activity_rmse[act] / count
            per_activity_rmse[act] = avg_rmse_act
            print(f"Activity {act}: RMSE = {avg_rmse_act:.4f}")

    total_samples_all = sum(total_activity_counts.values())
    overall_weighted_rmse = (
        sum(total_activity_rmse[act] for act in total_activity_rmse) / total_samples_all
        if total_samples_all > 0 else np.nan
    )

    print(f"\nOverall Weighted Average RMSE: {overall_weighted_rmse:.4f}")
    print("=" * 60)

    return 
