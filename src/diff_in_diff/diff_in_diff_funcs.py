import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
from statsmodels.formula.api import ols
from src.const import columns_to_predict, treatment_ids, placebo_ids
from src.get_data_funcs import get_infer_df, get_placebo_infer_df
from loguru import logger

def calc_diff_in_diff_per_target(
    infer_df: pd.DataFrame, 
    columns_to_predict: List[str], 
    treatment_auth_id: str, 
    control_auth_ids: List[str], 
    treatment_year: int,
    incident_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the difference in differences (DiD) for each column in columns_to_predict.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing data for multiple years.
    columns_to_predict (List[str]): The columns for which to calculate DiD.
    treatment_auth_id (str): The identifier for the treatment authority.
    control_auth_ids (List[str]): A list of identifiers for the control authorities.
    treatment_year (int): The year in which the treatment took place.
    results_dir (str): The directory to save the results.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the step 1 and step 2 results.
    """
    
    # Step 1: Calculate within-authority differences (1 year after vs 1 year before treatment year)
    step_1_results = []

    for auth_id in [treatment_auth_id] + control_auth_ids:
        auth_name = infer_df.loc[auth_id, 'auth_name'].unique().item()
        for col in columns_to_predict:
            before_treatment = infer_df[(infer_df.index == auth_id) & (infer_df['year'] == treatment_year - 1)][col].mean()
            after_treatment = infer_df[(infer_df.index == auth_id) & (infer_df['year'] == treatment_year + 1)][col].mean()
            before_after_diff = after_treatment - before_treatment

            step_1_results.append(pd.DataFrame([{
                'auth_id': auth_id,
                'auth_name': auth_name,
                'treatment_year': treatment_year,
                'incident_type': incident_type,
                'column': col,
                'before_after_diff': before_after_diff
            }]))
    
    step_1_results_df = pd.concat(step_1_results)
    
    # Step 2: Calculate difference in differences (DiD) between treatment and control authorities
    step_2_results = []

    treatment_results = step_1_results_df[step_1_results_df['auth_id'] == treatment_auth_id]
    target_auth_name = infer_df.loc[treatment_auth_id, 'auth_name'].unique().item()

    for control_auth_id in control_auth_ids:
        control_results = step_1_results_df[step_1_results_df['auth_id'] == control_auth_id]
        control_auth_name = infer_df.loc[control_auth_id, 'auth_name'].unique().item()
        for col in columns_to_predict:
            treatment_diff = treatment_results[treatment_results['column'] == col]['before_after_diff'].values[0]
            control_diff = control_results[control_results['column'] == col]['before_after_diff'].values[0]
            diff_in_diff = treatment_diff - control_diff
            
            step_2_results.append(pd.DataFrame([{
                'target_auth_id': treatment_auth_id,
                'target_auth_name': target_auth_name,
                'control_auth_id': control_auth_id,
                'control_auth_name': control_auth_name,
                'treatment_year': treatment_year,
                'incident_type': incident_type,
                'column': col,
                'diff_in_diff': diff_in_diff
            }]))
    
    step_2_results_df = pd.concat(step_2_results)
    return step_1_results_df, step_2_results_df

def plot_simple_did(
    diff_in_diff_df: pd.DataFrame, 
    k: int, 
    distance_metric: str,
    incident_type: str, 
    results_dir: str,
) -> None:
    """
    For each target authority and metric in the diff_in_diff_df:
    1. Plot the diff in diff values.
        - X-axis: auth_name
        - Y-axis: diff_in_diff
        - Color: green for treatment_auth_name, grey for all others
        - Add horizontal lines for mean and median diff_in_diff across control authorities.
        - Mention in the subtitle the distance metric of matching and k of nn.
    2. create results df.
        - Columns: target_auth_id, target_auth_name, control_auth_ids, control_auth_names,
            k, distance_metric, metric, mean_DiD, median_DiD, min_DiD, max_DiD.
    """
    # Iterate over each target authority and metric
    for (target_id, metric), group in diff_in_diff_df.groupby(['target_auth_id', 'column']):
        target_auth_name = group['target_auth_name'].iloc[0]

        # Calculate summary statistics
        mean_DiD = group['diff_in_diff'].mean()
        median_DiD = group['diff_in_diff'].median()

        # Plotting
        plt.figure(figsize=(10, 6))
        group = group.sort_values(by='control_auth_name')
        # Reverse the Hebrew text label for each authority
        group['control_auth_name'] = group['control_auth_name'].apply(lambda x: x[::-1])
        plt.bar(group['control_auth_name'], group['diff_in_diff'], color='grey')
        plt.bar(target_auth_name, group[group['control_auth_name'] == target_auth_name]['diff_in_diff'], color='green')

        # Add horizontal lines for mean and median
        plt.axhline(y=mean_DiD, color='orange', linestyle='--', label=f'Mean DiD: {mean_DiD:.0f}')
        plt.axhline(y=median_DiD, color='orange', linestyle='-.', label=f'Median DiD: {median_DiD:.0f}')
        
        # Add title and labels
        plt.title(f'Difference in Differences for {metric}', fontsize=16)
        plt.xticks(rotation=45)
        plt.xlabel('Authority Name', fontsize=14)
        plt.ylabel('Difference in Differences', fontsize=14)
        plt.legend()

        # Add subtitle with distance metric and k
        plt.suptitle(f'Distance Metric: {distance_metric}, k: {k}', fontsize=10)

        # Adjust layout to prevent cutting off
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save the plot
        plot_filename = f'DiD_plot_{target_id}_{distance_metric}_k{k}.png'
        plt.savefig(results_dir / metric / plot_filename)
        plt.close()

        print(f"Plot saved as {plot_filename}")
        
def plot_y_by_year(
    infer_df: pd.DataFrame, 
    target_id: str, 
    control_auth_ids: List[str], 
    columns_to_predict: List[str],
    diff_results_dir: str,
    k: int, 
    distance_metric: str,
    ):
    """
    Plot the values of the columns to predict for the target authority and its neighbors over the years.

    Args:
        df (pd.DataFrame): DataFrame containing data to plot.
        target_id (str): ID of the target authority.
        control_auth_ids (List[str]): List of control authority IDs.
        columns_to_predict (List[str]): List of columns to predict and plot.
        diff_results_dir (str): Directory to save the plot results.
        k (int): Number of nearest neighbors.
        distance_metric (str): Distance metric used for nearest neighbors.
    """
    
    # Reverse the Hebrew text label for each authority in the df
    infer_df['auth_name'] = infer_df['auth_name'].apply(lambda x: x[::-1])
    incident_df = infer_df[infer_df['auth_id'] == target_id][['incident_year', 'incident_type']].drop_duplicates()
    target_data = infer_df[infer_df['auth_id'] == target_id]
    all_control_data = infer_df[infer_df['auth_id'].isin(control_auth_ids)]
    control_dfs = [infer_df[infer_df['auth_id'] == auth_id] for auth_id in control_auth_ids]
    
    # Define a list of colors excluding green
    colors = ['#1c96f5', '#1ccef5', '#1be5c7', '#31d38e', '#3b8062', '#406455', '#7b8d86', '#63629c', '#8446af', '#cd6dcd']
    
    # Plot for each column to predict
    for column in columns_to_predict:
        plt.figure(figsize=(12, 8))

        # Plot the target authority
        plt.plot(target_data['year'], target_data[column], label=target_data['auth_name'].iloc[0], color='#f5ce1c', linewidth=3)

        # Plot each control authority in a different color
        for i, control_df in enumerate(control_dfs):
            color = colors[i % len(colors)]  # Cycle through colors if there are more control authorities than colors
            plt.plot(control_df['year'], control_df[column], label=control_df['auth_name'].iloc[0], color=color)

        # Add vertical lines and annotations for each incident_year and incident_type combination
        for _, row in incident_df.iterrows():
            incident_year = row['incident_year']
            incident_type = row['incident_type']
            
            plt.axvline(x=incident_year, color='grey', linestyle='--')
            plt.text(
                incident_year, 
                plt.ylim()[1], 
                incident_type,
                rotation=90, 
                verticalalignment='top',
                fontsize=10,
                color='black'
            )
        
        # Add title, subtitle, labels, and legend
        target_name = target_data['auth_name'].iloc[0]
        plt.title(f'{column} over Years | Target Auth = {target_name}', fontsize=16)
        plt.suptitle(f'Distance Metric: {distance_metric}, k: {k}', fontsize=10)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel(column, fontsize=14)
        plt.legend(title='Authority Name', loc='upper left')
        
        # Save the plot
        # make dir if not exists
        col_results_dir = Path(diff_results_dir/column)
        col_results_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = f'plot_{target_id}_distance_{distance_metric}_k{k}.png'
        plt.tight_layout()
        plt.savefig(col_results_dir / plot_filename)
        plt.close()
        
        ## Plot the avg controls against the target
        plt.figure(figsize=(12, 8))
        plt.plot(target_data['year'], target_data[column], label=target_data['auth_name'].iloc[0], color='#f5ce1c', linewidth=3)
        avg_control_data = all_control_data.groupby('year')[column].mean().reset_index()
        plt.plot(avg_control_data['year'], avg_control_data[column], label='Avg Control', color='grey', linestyle='--', linewidth=3)
        
        # Add vertical lines and annotations for each incident_year and incident_type combination
        for _, row in incident_df.iterrows():
            incident_year = row['incident_year']
            incident_type = row['incident_type']
            
            plt.axvline(x=incident_year, color='grey', linestyle='--')
            plt.text(
                incident_year, 
                plt.ylim()[1], 
                incident_type,
                rotation=90, 
                verticalalignment='top',
                fontsize=10,
                color='black'
            )
        
        # Add title, subtitle, labels, and legend
        target_name = target_data['auth_name'].iloc[0]
        plt.title(f'{column} over Years | Target Auth = {target_name}', fontsize=16)
        plt.suptitle(f'Distance Metric: {distance_metric}, k: {k}', fontsize=10)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel(column, fontsize=14)
        plt.legend(title='Authority Name', loc='upper left')
        
        # Save the plot
        # make dir if not exists
        col_results_dir = Path(diff_results_dir/column)
        col_results_dir.mkdir(parents=True, exist_ok=True)
        plot_filename = f'plot_{target_id}_avg_control_distance_{distance_metric}_k{k}.png'
        plt.tight_layout()
        plt.savefig(col_results_dir / plot_filename)
        plt.close()
    
def plot_y_by_year_for_all(    
    infer_df: pd.DataFrame, 
    columns_to_predict: List[str], 
    treatment_ids: List[str],
    k: int,
    distance_metric: str,
    results_dir: str,
    diff_results_dir: str,
    ):
    """
    Plot the values of the columns to predict for the target authority and its neighbors over the years.
    
    Parameters:
    infer_df (pd.DataFrame): The input DataFrame containing data for multiple years.
    columns_to_predict (List[str]): The columns for which to calculate DiD.
    treatment_ids (List[str]): The identifiers for the treatment authorities.
    k (int): number of nearest neighbors
    distance_metric (str): the distance metric used for matching
    results_dir (str): The directory to save the results.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the step 1 and step 2 results.
    """
    knn_results_path = results_dir / "knn" / f'knn_results_k={k}_{distance_metric}.csv'
    knn_results = pd.read_csv(knn_results_path)

    for target_id in tqdm(treatment_ids):
        control_auth_ids = knn_results[(knn_results['target_id'] == target_id) & (knn_results['is_nn'])]['control_id'].to_list()
        assert len(control_auth_ids) == k
        plot_y_by_year(infer_df, target_id, control_auth_ids, columns_to_predict, diff_results_dir, k, distance_metric)

def calc_simple_did(
    infer_df: pd.DataFrame, 
    columns_to_predict: List[str], 
    treatment_ids: List[str],
    k: int,
    distance_metric: str,
    results_dir: str,
    diff_results_dir: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate the difference in differences (DiD) for each column in columns_to_predict.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing data for multiple years.
    columns_to_predict (List[str]): The columns for which to calculate DiD.
    treatment_ids (List[str]): The identifiers for the treatment authorities.
    k (int): number of nearest neighbors
    distance_metric (str): the distance metric used for matching
    results_dir (str): The directory to save the results.
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames containing the step 1 and step 2 results.
    """
    infer_df.set_index('auth_id', inplace=True)
    knn_results_path = results_dir / "knn" / f'knn_results_k={k}_{distance_metric}.csv'
    knn_results = pd.read_csv(knn_results_path)
    
    all_step1_results = []
    all_step2_results = []
    all_compare_results = []
    # iterate over targets
    for target_id in tqdm(treatment_ids):
        control_auth_ids = knn_results[(knn_results['target_id'] == target_id) & (knn_results['is_nn'])]['control_id'].to_list()
        assert len(control_auth_ids) == k
        
        for (treatment_year, incident_type), _ in infer_df[infer_df.index == target_id].groupby(['incident_year', 'incident_type']):
            curr_df = infer_df[(infer_df['incident_type'] == incident_type) | (infer_df['incident_type'].isnull())]
            step_1_df, step_2_df = calc_diff_in_diff_per_target(curr_df, columns_to_predict, target_id, control_auth_ids, treatment_year, incident_type)
            plot_simple_did(step_2_df, k, distance_metric, incident_type, diff_results_dir)
            all_step1_results.append(step_1_df)
            all_step2_results.append(step_2_df)
    
    # Save the updated compare_df
    for results_list, file_name in zip([all_step1_results, all_step2_results, all_compare_results], ['step_1', 'step_2', 'compare']):
        results_filename = f'{file_name}_based_on_k={k}_{distance_metric}.csv'
        concat_df = pd.concat(results_list, ignore_index=True)
        concat_df.to_csv(diff_results_dir / results_filename, index=False)
        print(f"DF saved to {results_filename}")

def generate_data(n_units: int, n_timepoints: int, treatment_effect: float = 0, autocorrelation: float = 0.5)->pd.DataFrame:
    """
    Generate synthetic panel data with treatment effect and autocorrelation.

    Args:
        n_units (int): number of units
        n_timepoints (int): number of timepoints
        treatment_effect (float, optional): the size of the treatment effect. Defaults to 0.
        autocorrelation (float, optional): the strength of the autocorrelation. Defaults to 0.5.

    Returns:
        pd.DataFrame: _description_
    """
    units = np.repeat(np.arange(n_units), n_timepoints)
    time = np.tile(np.arange(n_timepoints), n_units)
    is_treatment = np.where(units >= n_units // 2, 1, 0)
    is_post = np.where(time >= n_timepoints // 2, 1, 0)

    # Simulate outcome with treatment effect and autocorrelation
    np.random.seed()  # Ensure variability across simulations
    error = np.random.randn(n_units * n_timepoints)
    for i in range(1, len(error)):
        if units[i] == units[i - 1]:
            error[i] += autocorrelation * error[i - 1]

    y = 2 + 0.5 * is_treatment + 1 * is_post + treatment_effect * is_treatment * is_post + error
    data = pd.DataFrame({
        'Y': y,
        'IsTreatment': is_treatment,
        'IsPost': is_post,
        'Unit': units,
        'time': time
    })
    return data

def plot_data(data: pd.DataFrame, n_units: int, n_timepoints: int, results_dir: str, filename: str):
    """
    Plot the outcome over time for each unit, with a vertical line at the treatment time.

    Args:
        data (pd.DataFrame): The input DataFrame containing the outcome data.
        n_units (int): number of units
        n_timepoints (int): number of timepoints
        results_dir (str): The directory to save the results.
        filename (str): The filename for the plot.
    """
    plt.figure(figsize=(10, 6))
    units = data['Unit'].unique()
    
    # Identify treatment and control units
    treatment_units = units[units >= n_units // 2]
    control_units = units[units < n_units // 2]
    
    # Define colors for control units using the provided palette
    colors = [
        '#1c96f5', '#1ccef5', '#1be5c7', '#31d38e', '#3b8062',
        '#406455', '#7b8d86', '#63629c', '#8446af', '#cd6dcd'
    ]
    
    # Map each control unit to a color from the palette
    color_map = {}
    for idx, unit in enumerate(control_units):
        color_map[unit] = colors[idx % len(colors)]
    
    # Plot each unit's data
    for unit in units:
        unit_data = data[data['Unit'] == unit]
        if unit in treatment_units:
            plt.plot(
                unit_data['time'], unit_data['Y'],
                label=f'Unit {unit}',
                color='#f5ce1c', linewidth=3)
        else:
            plt.plot(
                unit_data['time'], unit_data['Y'],
                label=f'Unit {unit}',
                color=color_map[unit])
    
    # Add dashed vertical line at treatment time with label
    treatment_time = n_timepoints // 2
    plt.axvline(x=treatment_time, color='k', linestyle='--')
    plt.text(
        treatment_time + 0.1, plt.ylim()[1] * 0.95, 'Treatment',
        rotation=90, verticalalignment='top')
    
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.title('Outcome over Time by Unit')
    # Optionally, include legend if needed
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(results_dir / filename)
    plt.close()

def compute_did(data: pd.DataFrame)->Tuple[float, float, float]:
    """
    Compute the difference-in-differences (DiD) estimate and p value.

    Args:
        data (pd.DataFrame): The input DataFrame containing the outcome data.

    Returns:
        Tuple[float, float, float]: The DiD estimate, standard error, and p value.
    """
    
    model = ols('Y ~ IsTreatment * IsPost', data=data).fit()
    coef = model.params['IsTreatment:IsPost']
    se = model.bse['IsTreatment:IsPost']
    pvalue = model.pvalues['IsTreatment:IsPost']
    return coef, se, pvalue

def compute_did_fixed_effects(data: pd.DataFrame)->Tuple[float, float, float]:
    """
    Compute the difference-in-differences (DiD) estimate with fixed effects.

    Args:
        data (pd.DataFrame): The input DataFrame containing the outcome data.

    Returns:
        Tuple[float, float, float]: The DiD estimate, standard error, and p value.
    """
    model_fe = ols('Y ~ IsTreatment * IsPost + C(Unit) + C(time)', data=data).fit()
    coef = model_fe.params['IsTreatment:IsPost']
    se = model_fe.bse['IsTreatment:IsPost']
    pvalue = model_fe.pvalues['IsTreatment:IsPost']
    return coef, se, pvalue

def compute_clustered_did(data: pd.DataFrame)->Tuple[float, float, float]:
    """
    Compute the difference-in-differences (DiD) estimate with clustered standard errors.

    Args:
        data (pd.DataFrame): The input DataFrame containing the outcome data.

    Returns:
        Tuple[float, float, float]: The DiD estimate, standard error, and p value.
    """
    model = ols('Y ~ IsTreatment * IsPost', data=data).fit()
    clustered_model = model.get_robustcov_results(cov_type='cluster', groups=data['Unit'])
    
    # Convert NumPy arrays to pandas Series with index labels
    params = pd.Series(clustered_model.params, index=model.params.index)
    bse = pd.Series(clustered_model.bse, index=model.bse.index)
    pvalues = pd.Series(clustered_model.pvalues, index=model.pvalues.index)
    
    coef = params['IsTreatment:IsPost']
    se = bse['IsTreatment:IsPost']
    pvalue = pvalues['IsTreatment:IsPost']
    return coef, se, pvalue

def permutation_test(data: pd.DataFrame, n_treatment_units: int, n_permutations=100)->float:
    """
    Perform a permutation test to compute the p-value for the DiD estimate.

    Args:
        data (pd.DataFrame): The input DataFrame containing the outcome data.
        n_permutations (int, optional): The number of permutations to perform. Defaults to 100.

    Returns:
        float: The p-value for the permutation test.
    """
    units = data['Unit'].unique()
    # Original estimate
    coef_original, _, _ = compute_did(data)
    permuted_did = []

    for _ in range(n_permutations):
        # Randomly assign treatment to units
        shuffled_units = np.random.permutation(units)
        treatment_units = shuffled_units[:n_treatment_units]
        # Create a mapping from units to IsTreatment
        unit_treatment_mapping = {unit: 1 if unit in treatment_units else 0 for unit in units}
        # Apply the shuffled IsTreatment to the data
        shuffled_data = data.copy()
        shuffled_data['IsTreatment'] = shuffled_data['Unit'].map(unit_treatment_mapping)
        coef_perm, _, _ = compute_did(shuffled_data)
        permuted_did.append(coef_perm)

    p_value = np.mean([abs(did) >= abs(coef_original) for did in permuted_did])
    return p_value

def plot_results(results_dict: dict, results_dir: str, filename: str):
    """
    Plot the results of the method comparison.

    Args:
        results_df (dict): The DataFrame containing the method comparison results.
        results_dir (str): The directory to save the results.
        filename (str): The filename for the plot.
    """
    methods = results_dict['method']
    fpr = [results_dict['false_positive_rate'][method] for method in methods]
    power = [results_dict['power'][method] for method in methods]

    plt.figure(figsize=(8, 6))
    _ = plt.scatter(power, fpr, c=range(len(methods)), cmap='viridis', s=100)

    # Add method labels
    for i, method in enumerate(methods):
        plt.text(power[i], fpr[i], method, fontsize=9, ha='right')

    # Add x-lines at 0, 0.8 (red), and 1
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=0.8, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)

    # Add y-lines at 0 and 1
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Power')
    plt.ylabel('False Positive Rate')
    plt.title('Method Comparison')
    plt.tight_layout()

    # Save the figure to the specified directory
    plt.savefig(results_dir / filename)
    plt.close()

def compare_methods_using_sim_data(n_simulations: int, n_units: int, n_timepoints: int, treatment_effect: float, results_dir: str)->dict:
    """
    Compare different methods for estimating the treatment effect using simulated data.

    Args:
        n_simulations (int): The number of simulations to run.
        n_units (int): The number of units in the data.
        n_timepoints (int): The number of timepoints in the data.
        treatment_effect (float): The true treatment effect.
        results_dir (str): The directory to save the results.

    Returns:
        dict: A dictionary containing the results of the method comparison.
    """
    methods = ['OLS', 'OLS + FE', 'Clustered SE', 'Permutation']
    results = {
        'method': methods,
        'false_positive_rate': {},
        'power': {}
    }

    # Counters for false positives and power
    false_positive_count = {method: 0 for method in methods}
    power_count = {method: 0 for method in methods}

    # Run simulations
    for sim in tqdm(range(n_simulations), desc="Simulations Progress"):
        # Generate data with no treatment effect for false positive tests
        data_no_effect = generate_data(n_units, n_timepoints, treatment_effect=0)
        if sim == 0:
            plot_data(data_no_effect, n_units, n_timepoints, results_dir, f'data_no_effect_{n_units=}.png')

        # Generate data with treatment effect for power tests
        data_with_effect = generate_data(n_units, n_timepoints, treatment_effect=treatment_effect)
        if sim == 0:
            plot_data(data_with_effect, n_units, n_timepoints, results_dir, f'data_with_effect_{n_units=}.png')

        # OLS method
        _, _, pvalue_ols_no_effect = compute_did(data_no_effect)
        _, _, pvalue_ols_with_effect = compute_did(data_with_effect)
        if pvalue_ols_no_effect < 0.05:
            false_positive_count['OLS'] += 1
        if pvalue_ols_with_effect < 0.05:
            power_count['OLS'] += 1

        # Fixed Effects method 
        _, _, pvalue_fe_no_effect = compute_did_fixed_effects(data_no_effect)
        _, _, pvalue_fe_with_effect = compute_did_fixed_effects(data_with_effect)
        if pvalue_fe_no_effect < 0.05:
            false_positive_count['OLS + FE'] += 1
        if pvalue_fe_with_effect < 0.05:
            power_count['OLS + FE'] += 1

        # Clustered Standard Errors
        _, _, pvalue_cluster_no_effect = compute_clustered_did(data_no_effect)
        _, _, pvalue_cluster_with_effect = compute_clustered_did(data_with_effect)
        if pvalue_cluster_no_effect < 0.05:
            false_positive_count['Clustered SE'] += 1
        if pvalue_cluster_with_effect < 0.05:
            power_count['Clustered SE'] += 1

        # Permutation test
        p_value_no_effect = permutation_test(data_no_effect, n_treatment_units=n_units // 2)
        p_value_with_effect = permutation_test(data_with_effect, n_treatment_units=n_units // 2)
        if p_value_no_effect < 0.05:
            false_positive_count['Permutation'] += 1
        if p_value_with_effect < 0.05:
            power_count['Permutation'] += 1

    # Calculate false positive rates and power
    for method in methods:
        results['false_positive_rate'][method] = false_positive_count[method] / n_simulations
        results['power'][method] = power_count[method] / n_simulations

    # Print results
    print(f"False Positive Rates: {results['false_positive_rate']}")
    print(f"Power: {results['power']}")

    return results

def run_simulation_with_did_comparison():
    # Run the simulation and comparison
    n_simulations = 100  # Number of simulations for statistical reliability
    n_timepoints = 8
    treatment_effect = 1  # Adjust based on desired effect size
    
    src_path = Path.cwd() / 'src'
    print(src_path)
    results_dir = src_path / 'results' / 'simulation_did'
    
    for n_units in [10, 30, 50]:
        results = compare_methods_using_sim_data(n_simulations, n_units, n_timepoints, treatment_effect, results_dir)
        plot_results(results, results_dir, f'methods_comparison_plot_n_units={n_units}.png')

def test_simple_did():
    # Example DataFrame
    inference_df = pd.DataFrame({
        'auth_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7],
        'auth_name': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F', 'G', 'G', 'G', 'G'],
        'year': [2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002, 2000, 2002],
        'metric1': [10, 15, 12, 15, 9, 10, 14, 19, 8, 19, 11, 21, 10, 12, 9, 13],
        'metric2': [100, 150, 120, 180, 90, 130, 140, 190, 80, 120, 110, 120, 100, 130, 90, 140],
        'incident_type': [
            None, None, None, None, None, None, None, None, None, None, 
            'conviction', 'conviction', 'conviction', 'conviction', 'arrest', 'arrest'],
        'incident_year': [None, None, None, None, None, None, None, None, None, None, 2001, 2001, 2001, 2001, 2001, 2001]
    })
    inference_df.set_index('auth_id', inplace=True)
    
    # New data for 2004
    new_data = pd.DataFrame({
        'auth_id': [1, 2, 3, 4, 5, 6, 7],
        'auth_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        'year': [2004] * 7,
        'metric1': [18, 17, 11, 20, 22, 25, 14],
        'metric2': [160, 200, 140, 210, 130, 140, 180],
        'incident_type': [None, None, None, None, None, None, 'conviction'],
        'incident_year': [None, None, None, None, None, None, 2003]
    })

    # Set index and append the new data to inference_df
    new_data.set_index('auth_id', inplace=True)
    inference_df = pd.concat([inference_df, new_data])

    df_treatment = pd.DataFrame({
        'auth_id': [6, 7],
        'auth_name': ['F', 'G'],
        'feature1': [0.6, 0.7],
        'feature2': [0.1, 1.2],
        'feature3': [0.7, 0.5],
    }) 
    
    src_path = Path.cwd() / 'src'
    print(src_path)
    results_dir = src_path / 'results' / 'test'
    
    calc_simple_did(
        inference_df,
        columns_to_predict=['metric1', 'metric2'], 
        treatment_ids=df_treatment['auth_id'].tolist(),
        k=3,
        distance_metric='cosine',
        results_dir=results_dir,
        diff_results_dir=(results_dir / 'diff'),
    )
    
def run_did_analysis_per_target(data: pd.DataFrame, imputed: bool):
    p_vals = {}
    
    # OLS method
    _, _, pvalue_ols = compute_did(data)
    p_vals['OLS'] = pvalue_ols

    # Fixed Effects method 
    _, _, pvalue_fe = compute_did_fixed_effects(data)
    p_vals['OLS + FE'] = pvalue_fe

    # Clustered Standard Errors
    if imputed:
        _, _, pvalue_cluster = compute_clustered_did(data)
        p_vals['Clustered SE'] = pvalue_cluster
    else:
        p_vals['Clustered SE'] = np.nan
    
    # Permutation test
    p_value_perm = permutation_test(data, n_treatment_units=1)
    p_vals['Permutation'] = p_value_perm
    
    return p_vals

def get_data_by_target_and_incident(infer_df, target_id, incident_year, incident_type, control_ids, col_to_predict):
    # returned df should contain the columns: IsTreatment, IsPost, Unit, time, Y
    # Unit: auth_id
    # time: year
    # IsTreatment: 1 if the unit is a treatment unit, 0 otherwise
    # IsPost: 1 if the time is after the treatment, 0 otherwise
    # Y: the value of the metric to predict
    
    # filter by id
    filtered_df = infer_df[(infer_df['auth_id'] == target_id) | (infer_df['auth_id'].isin(control_ids))].reset_index(drop=True)
    # filter by incident
    filtered_df = filtered_df[
        ((filtered_df['incident_year'] == incident_year) | (filtered_df['incident_year'].isna())) 
        & ((filtered_df['incident_type'] == incident_type) | (filtered_df['incident_type'].isna()))].reset_index(drop=True)
    # add IsTreatment
    filtered_df['IsTreatment'] = np.where(filtered_df['auth_id'] == target_id, 1, 0)
    # add IsPost
    filtered_df['IsPost'] = np.where(filtered_df['year'] > incident_year, 1, 0)
    # select relevant columns
    filtered_df = filtered_df[['auth_id', 'year', 'IsTreatment', 'IsPost', col_to_predict]]
    # rename columns
    filtered_df.rename(columns={col_to_predict: 'Y', 'auth_id': 'Unit', 'year': 'time'}, inplace=True)
    return filtered_df

def validate_enough_data_for_inference(df: pd.DataFrame, incident_year: int, col_to_predict: str, years_range: int = 1, n_years_before: int = 3, n_years_after: int = 1):
    """
    validate enough values for col_to_predict

    Args:
        df (_type_): The DataFrame containing the data.
        incident_year (_type_): The year of the incident.
        col_to_predict (_type_): The column to predict.
        years_range (int, optional): The range of years that require values. Defaults to 1.
        n_years_before (int, optional): The min number of years before the incident year. Defaults to 3.
        n_years_after (int, optional): The min number of years on / after the incident year. Defaults to 2.

    Returns:
        bool: True if there are enough values for inference, False otherwise.
    """
    # check that there are values for the predict column for all years in the range [incident_year-years_range, incident_year+years_range]
    df_relevant_years = df[df.year.isin(list(range(incident_year-years_range, incident_year+years_range)))]
    if df_relevant_years[col_to_predict].isna().any():
        return False
    
    # check that there are at least n_years_before with not null values in col_to_predict before the incident year
    df_before = df[(df['year'] < incident_year) & (df[col_to_predict].notna())]
    if len(df_before) < n_years_before:
        return False    
    
    # check that there are at least n_years_after values after the incident year
    df_after = df[(df['year'] >= incident_year) & (df[col_to_predict].notna())]
    if len(df_after) < n_years_after:
        return False
    
    # valid df for inference
    return True

def run_did_analysis(
        infer_df: pd.DataFrame, 
        columns_to_predict: List[str], 
        treatment_ids: List[str], 
        k: int, 
        distance_metric: str, 
        results_dir: str, 
        diff_results_dir: bool, 
        imputed: bool, 
        placebo: bool
    ):
    # placebo params
    if placebo:
        treatment_ids = placebo_ids
        infer_df = get_placebo_infer_df(imputed, src_path)
    
    knn_results_path = results_dir / "knn" / f'knn_results_k={k}_{distance_metric}_placebo={placebo}.csv'
    knn_results = pd.read_csv(knn_results_path)
    incident_df = infer_df[['auth_id', 'incident_year', 'incident_type']].drop_duplicates()
    
    # iterate over columns to predict
    for col_to_predict in columns_to_predict:
        logger.info(f"Run | DiD analysis for column: {col_to_predict}")
        results = []
        
        # iterate over targets
        for target_id in tqdm(treatment_ids):
            # get control ids
            control_ids = knn_results[(knn_results['target_id'] == target_id) & (knn_results['is_nn'])]['control_id'].to_list()
            assert len(control_ids) == k
            
            # iterate over incidents
            curr_incident_df = incident_df[incident_df['auth_id'] == target_id]
            for _, row in curr_incident_df.iterrows():
                incident_year = row['incident_year']
                incident_type = row['incident_type']
                
                # filter data
                target_df = get_data_by_target_and_incident(infer_df, target_id, incident_year, incident_type, control_ids, col_to_predict)
                # validate_enough_data
                enough_data = validate_enough_data_for_inference(target_df, incident_year, col_to_predict)
                if not enough_data:
                    logger.info(f'not enough data for target {target_id} in year {incident_year}, incident_type {incident_type}')
                    continue
                
                # run did analysis
                p_vals = run_did_analysis_per_target(target_df, imputed)
                # append results
                target_name = infer_df[infer_df['auth_id'] == target_id]['auth_name'].iloc[0]
                results.append(pd.DataFrame([{
                    'target_id': target_id,
                    'target_name': target_name,
                    'incident_year': incident_year,
                    'incident_type': incident_type,
                    'column': col_to_predict,
                    'p_vals': p_vals,
                    'n_rows_data': len(target_df),
                    'n_rows_notna': target_df['Y'].notna().sum()
                }]))
                
        results_df = pd.concat(results, ignore_index=True)
        # unpack p_vals dict
        results_df = pd.concat([results_df.drop(['p_vals'], axis=1), results_df['p_vals'].apply(pd.Series)], axis=1)
        results_df.to_csv(diff_results_dir / col_to_predict / f'did_results_placebo={placebo}_imputed={imputed}_k={k}_{distance_metric}.csv', index=False)


if __name__ == "__main__":
    # ----- Simulated Data -----
    # test_simple_did()
    # run_simulation_with_did_comparison()
    
    # ----- Our Data -----

    src_path = Path.cwd() / 'src'
    results_dir = src_path / 'results'
    diff_results_dir = results_dir / 'diff'
    imputed = False
    placebo = False
    infer_df = get_infer_df(imputed)
    
    # plot_y_by_year_for_all(        
    #     infer_df,
    #     columns_to_predict=columns_to_predict, 
    #     treatment_ids=treatment_ids,
    #     k=10,
    #     distance_metric='cosine',
    #     results_dir=results_dir,
    #     diff_results_dir=diff_results_dir
    # )
    
    run_did_analysis(
        infer_df,
        columns_to_predict=columns_to_predict, 
        treatment_ids=treatment_ids,
        k=10,
        distance_metric='cosine',
        results_dir=results_dir,
        diff_results_dir=diff_results_dir,
        imputed=imputed,
        placebo=placebo
    )