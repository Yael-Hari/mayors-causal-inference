import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.linear_model import LinearRegression

def calc_diff_in_diff(
    df: pd.DataFrame, 
    columns_to_predict: List[str], 
    treatment_auth_code: str, 
    control_auth_codes: List[str], 
    treatment_year: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the difference in differences (DiD) for each column in columns_to_predict.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing data for multiple years.
    columns_to_predict (List[str]): The columns for which to calculate DiD.
    treatment_auth_code (str): The identifier for the treatment authority.
    control_auth_codes (List[str]): A list of identifiers for the control authorities.
    treatment_year (int): The year in which the treatment took place.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the step 1 and step 2 results.
    """
    
    # Step 1: Calculate within-authority differences (1 year after vs 1 year before treatment year)
    step_1_results = []

    for auth_code in [treatment_auth_code] + control_auth_codes:
        auth_name = df.loc[auth_code, 'auth_name']
        for col in columns_to_predict:
            before_treatment = df[(df.index == auth_code) & (df['year'] == treatment_year - 1)][col].mean()
            after_treatment = df[(df.index == auth_code) & (df['year'] == treatment_year + 1)][col].mean()
            before_after_diff = after_treatment - before_treatment

            step_1_results.append({
                'auth_code': auth_code,
                'auth_name': auth_name,
                'treatment_year': treatment_year,
                'column': col,
                'before_after_diff': before_after_diff
            })
    
    step_1_results_df = pd.DataFrame(step_1_results)
    
    # Step 2: Calculate difference in differences (DiD) between treatment and control authorities
    step_2_results = []

    treatment_results = step_1_results_df[step_1_results_df['auth_code'] == treatment_auth_code]

    for control_auth_code in control_auth_codes:
        control_results = step_1_results_df[step_1_results_df['auth_code'] == control_auth_code]
        for col in columns_to_predict:
            treatment_diff = treatment_results[treatment_results['column'] == col]['before_after_diff'].values[0]
            control_diff = control_results[control_results['column'] == col]['before_after_diff'].values[0]
            diff_in_diff = treatment_diff - control_diff
            
            step_2_results.append({
                'target_auth_code': treatment_auth_code,
                'target_auth_name': df.loc[treatment_auth_code, 'auth_name'],
                'control_auth_code': control_auth_code,
                'control_auth_name': df.loc[control_auth_code, 'auth_name'],
                'treatment_year': treatment_year,
                'column': col,
                'diff_in_diff': diff_in_diff
            })
    
    step_2_results_df = pd.DataFrame(step_2_results)

    # Save the results
    step_1_results_filename = f'step_1_results_{treatment_auth_code}_{treatment_year}.csv'
    step_1_results_df.to_csv(step_1_results_filename, index=False)
    
    step_2_results_filename = f'step_2_results_{treatment_auth_code}_{treatment_year}.csv'
    step_2_results_df.to_csv(step_2_results_filename, index=False)
    
    print(f"Step 1 results saved as {step_1_results_filename}")
    print(f"Step 2 results saved as {step_2_results_filename}")

    return step_1_results_df, step_2_results_df

def predict_metrics_for_target_auth(
    df: pd.DataFrame, 
    target_auth_codes: List[str], 
    cols_to_predict: List[str]
) -> pd.DataFrame:
    """
    Predict the metric for each target authority in the year after the treatment year
    using previous values from years before the treatment year.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing data for multiple years.
    target_auth_codes (List[str]): List of authority codes to predict metrics for.
    cols_to_predict (List[str]): List of columns to predict.

    Returns:
    pd.DataFrame: DataFrame containing the predictions for each target authority and column.
    """

    results = []

    for auth_code in target_auth_codes:
        # Get the treatment year for the target authority
        treatment_year = df.loc[auth_code, 'treatment_year']

        for col in cols_to_predict:
            # Prepare data for prediction: years before the treatment year
            data = df[(df.index == auth_code) & (df['year'] < treatment_year)][['year', col]].dropna()

            if len(data) > 1:  # Ensure there is enough data to fit a model
                # Train a linear regression model
                X = data[['year']]
                y = data[col]
                model = LinearRegression()
                model.fit(X, y)

                # Predict the metric for the year after the treatment year
                prediction_year = treatment_year + 1
                predicted_value = model.predict([[prediction_year]])[0]

                results.append({
                    'auth_code': auth_code,
                    'auth_name': df.loc[auth_code, 'auth_name'],
                    'treatment_year': treatment_year,
                    'prediction_year': prediction_year,
                    'column': col,
                    'predicted_value': predicted_value
                })
            else:
                print(f"Not enough data to predict {col} for authority {auth_code}. Skipping.")
    
    # Convert results to a DataFrame
    predictions_df = pd.DataFrame(results)
    
    # Save the results
    predictions_filename = 'predicted_metrics.csv'
    predictions_df.to_csv(predictions_filename, index=False)
    
    print(f"Predictions saved as {predictions_filename}")
    
    return predictions_df

def compare_pred_true(
    diff_in_diff_df: pd.DataFrame, 
    k: int, 
    distance_metric: str, 
    compare_df_path: str
) -> None:
    """
    For each target authority and metric in the diff_in_diff_df:
    1. Plot the diff in diff values.
        - X-axis: auth_name
        - Y-axis: diff_in_diff
        - Color: green for treatment_auth_name, grey for all others
        - Add horizontal lines for mean and median diff_in_diff across control authorities.
        - Mention in the subtitle the distance metric of matching and k of nn.
    2. Update the compare_df in path compare_df_path with the results.
        - Columns: target_auth_code, target_auth_name, control_auth_codes, control_auth_names,
            k, distance_metric, metric, mean_DiD, median_DiD, min_DiD, max_DiD.
    """

    # Ensure the compare_df exists or create it
    try:
        compare_df = pd.read_csv(compare_df_path)
    except FileNotFoundError:
        compare_df = pd.DataFrame(columns=[
            'target_auth_code', 'target_auth_name', 'control_auth_codes', 'control_auth_names',
            'k', 'distance_metric', 'metric', 'mean_DiD', 'median_DiD', 'min_DiD', 'max_DiD'
        ])

    # Iterate over each target authority and metric
    for (target_auth_code, metric), group in diff_in_diff_df.groupby(['target_auth_code', 'column']):
        target_auth_name = group['target_auth_name'].iloc[0]
        control_auth_codes = group['control_auth_code'].unique().tolist()
        control_auth_names = group['control_auth_name'].unique().tolist()

        # Calculate summary statistics
        mean_DiD = group['diff_in_diff'].mean()
        median_DiD = group['diff_in_diff'].median()
        min_DiD = group['diff_in_diff'].min()
        max_DiD = group['diff_in_diff'].max()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(group['control_auth_name'], group['diff_in_diff'], color='grey')
        plt.bar(target_auth_name, group[group['control_auth_name'] == target_auth_name]['diff_in_diff'], color='green')

        # Add horizontal lines for mean and median
        plt.axhline(y=mean_DiD, color='blue', linestyle='--', label=f'Mean DiD: {mean_DiD:.2f}')
        plt.axhline(y=median_DiD, color='red', linestyle='-.', label=f'Median DiD: {median_DiD:.2f}')
        
        # Add title and labels
        plt.title(f'Difference in Differences for {metric}', fontsize=16)
        plt.xlabel('Authority Name', fontsize=14)
        plt.ylabel('Difference in Differences', fontsize=14)
        plt.legend()

        # Add subtitle with distance metric and k
        plt.suptitle(f'Distance Metric: {distance_metric}, k: {k}', fontsize=10, y=0.92)

        # Save the plot
        plot_filename = f'DiD_plot_{target_auth_code}_{metric}_{distance_metric}_k{k}.png'
        plt.savefig(plot_filename)
        plt.close()

        print(f"Plot saved as {plot_filename}")

        # Update the compare_df
        compare_df = compare_df.append({
            'target_auth_code': target_auth_code,
            'target_auth_name': target_auth_name,
            'control_auth_codes': ','.join(map(str, control_auth_codes)),
            'control_auth_names': ','.join(control_auth_names),
            'k': k,
            'distance_metric': distance_metric,
            'metric': metric,
            'mean_DiD': mean_DiD,
            'median_DiD': median_DiD,
            'min_DiD': min_DiD,
            'max_DiD': max_DiD
        }, ignore_index=True)

    # Save the updated compare_df
    compare_df.to_csv(compare_df_path, index=False)
    print(f"Comparison DataFrame updated and saved to {compare_df_path}")

if __name__ == "__main__":
    
    # Example DataFrame
    df = pd.DataFrame({
        'auth_code': [1, 1, 2, 2, 3, 3, 4, 4],
        'auth_name': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
        'year': [2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002],
        'metric1': [10, 15, 12, 18, 9, 13, 14, 20],
        'metric2': [100, 150, 120, 180, 90, 130, 140, 200]
    })
    df.set_index('auth_code', inplace=True)

    # Call the function
    step_1_df, step_2_df = calc_diff_in_diff(df, columns_to_predict=['metric1', 'metric2'], treatment_auth_code=1, control_auth_codes=[2, 3], treatment_year=2001)

    # Example DataFrame for diff_in_diff_df
    diff_in_diff_df = pd.DataFrame({
        'target_auth_code': [1, 1, 1, 2, 2, 2],
        'target_auth_name': ['A', 'A', 'A', 'B', 'B', 'B'],
        'control_auth_code': [3, 4, 5, 6, 7, 8],
        'control_auth_name': ['C', 'D', 'E', 'F', 'G', 'H'],
        'treatment_year': [2001, 2001, 2001, 2002, 2002, 2002],
        'column': ['metric1', 'metric1', 'metric1', 'metric2', 'metric2', 'metric2'],
        'diff_in_diff': [1.2, -0.8, 0.5, 1.1, -0.7, 0.6]
    })
    # Call the function
    compare_pred_true(diff_in_diff_df, k=3, distance_metric='euclidean', compare_df_path='compare_df.csv')