import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from tqdm import tqdm
from src.matching.create_vecs import create_vecs
from src.const import treatment_ids

def match_and_plot_per_target(matching_df: pd.DataFrame, matching_cols: List[str], k: int, target_id: int, results_dir: str, metric='euclidean'):
    """
    Find the k nearest neighbors for a specific authority, plot the results with dimensionality reduction,
    and save both the plot and the results DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    matching_cols (list): The columns to be used for matching.
    k (int): The number of nearest neighbors to find.
    target_id: The identifier of the target authority for which to find neighbors.
    results_dir: path to dir to save results.
    metric (str): The distance metric to be used in NearestNeighbors.
    
    Returns:
    pd.DataFrame: A DataFrame containing the k nearest neighbors for the target authority.
    """

    # Check for columns with null values and print them
    null_columns = matching_df[matching_cols].columns[matching_df[matching_cols].isnull().any()]
    if len(null_columns) > 0:
        print(f"Columns with null values: {list(null_columns)}")
        return None

    # Ensure the target authority exists in the DataFrame
    if target_id not in matching_df.index:
        print(f"Target authority '{target_id}' not found in the DataFrame.")
        return None
    
    # Fit the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric)
    nn.fit(matching_df[matching_cols])
    
    # Find the nearest neighbors for the target authority
    target_index = matching_df.index.get_loc(target_id)
    distances, indices = nn.kneighbors(matching_df[matching_cols].iloc[[target_index]])
    
    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(matching_df[matching_cols])
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot all points in grey
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='grey', label='Others')

    # Highlight the k nearest neighbors in yellow
    nn_indices = indices[0][1:]  # Skip the first index, which is the target itself
    plt.scatter(reduced_data[nn_indices, 0], reduced_data[nn_indices, 1], color='#f7d163', label=f'Top {k} Neighbors')

    # Highlight the target authority in green
    plt.scatter(reduced_data[target_index, 0], reduced_data[target_index, 1], color='green', label='Target Authority')
    
    # Add labels for all points
    for i in range(len(reduced_data)):
        # Reverse the Hebrew text label for each authority
        reversed_label = matching_df.iloc[i]['auth_name'][::-1]
        plt.text(reduced_data[i, 0] + 0.01, reduced_data[i, 1] + 0.01, reversed_label, fontsize=9)
        
    # Add title and subtitle
    target_name = matching_df.loc[target_id]['auth_name']
    plt.title('KNN for Target {}'.format(target_name[::-1]))
    # Reverse the Hebrew labels for the k-nearest neighbors
    knn_auth_names = ', '.join(matching_df.iloc[nn_indices].sort_values(by="auth_id")['auth_name'].apply(lambda x: x[::-1]))
    plt.suptitle(f'K Nearest Neighbors: {knn_auth_names}', fontsize=12)
    
    # Add legend
    plt.legend()

    # Save the plot
    plot_filename = f'id={target_id}_k={k}_{metric}.png'
    plt.savefig(results_dir / plot_filename)
    plt.close()

    print(f"Plot saved as {plot_filename}")

    # Prepare the results DataFrame
    results = []
    for i in range(len(matching_df)):
        results.append({
            'control_id': matching_df.index[i],
            'control_name': matching_df.iloc[i]['auth_name'],
            'target_id': target_id,
            'k': k,
            'is_nn': matching_df.index[i] in matching_df.index[nn_indices]
        })

    results_df = pd.DataFrame(results)
    return results_df

def match_and_plot(df_treatment: pd.DataFrame, df_control: pd.DataFrame, matching_cols: List[str],k: int, metric: str, results_dir: str, placebo: bool = False):
    """
    Match each target authority with the k nearest neighbors in the control group based on the specified columns.

    Args:
        df_treatment (pd.DataFrame): the treatment group DataFrame
        df_control (pd.DataFrame): the control group DataFrame
        matching_cols (List[str]): names of the columns to use for matching
        k (int): number of nearest neighbors to find
        metric (str): the distance metric to use for matching
    """
    # Call the function
    target_ids = df_treatment['auth_id'].unique()

    all_targets_results = []
    # iterate over targets
    for target_id in target_ids:
        target_row = df_treatment[df_treatment['auth_id'] == target_id]
        matching_df = pd.concat([df_control, target_row], axis=0)
        matching_df.set_index('auth_id', inplace=True)
        results_df = match_and_plot_per_target(matching_df, matching_cols=matching_cols, k=k, target_id=target_id, results_dir=results_dir, metric=metric)
        all_targets_results.append(results_df)
        print(results_df)
        
    # Save the results DataFrame
    results_filename = f'knn_results_k={k}_{metric}_placebo={placebo}.csv'
    all_targets_results = pd.concat(all_targets_results, ignore_index=True)
    all_targets_results.to_csv(results_dir / results_filename, index=False)
    
    print(f"Results DataFrame saved as {results_filename}")
    return all_targets_results

def test_match_and_plot():
    metric = 'euclidean' # 'euclidean' or 'cosine' or 'manhattan' or 'minkowski' or 'chebyshev' or 'hamming' or 'jaccard'

    # Example DataFrame
    df_control = pd.DataFrame({
        'auth_id': [1, 2, 3, 4, 5],
        'auth_name': ['A', 'B', 'C', 'D', 'E'],
        'feature1': [0.1, 0.3, 0.2, 0.5, 0.4],
        'feature2': [1.2, 0.9, 1.5, 1.0, 1.3],
        'feature3': [0.3, 0.2, 0.1, 0.6, 1.3]
    })
    
    df_treatment = pd.DataFrame({
        'auth_id': [6, 7, 8],
        'auth_name': ['F', 'G', 'H'],
        'feature1': [0.6, 0.7, 0.8],
        'feature2': [0.1, 1.2, 0.4],
        'feature3': [0.7, 0.5, 0.2]
    })    

    src_path = Path.cwd() / 'src'
    print(src_path)
    results_dir = src_path / 'results' / 'test'
    knn_results_dir = results_dir / 'knn'
    
    k = 3
    metric='cosine' # 'euclidean' or 'cosine' or 'manhattan' or 'minkowski' or 'chebyshev' or 'hamming' or 'jaccard'
    matching_cols=['feature1', 'feature2']
    match_and_plot(df_treatment, df_control, matching_cols, k, metric, knn_results_dir)

def run_matching(k, metric):
    src_path = Path.cwd() / 'src'

    df_matching = pd.read_csv(src_path / 'data/matching_features_df.csv')
    df_matching = df_matching.rename(columns={'authority_code': 'auth_id', 'authority_name': 'auth_name'})
    df_treatment = df_matching[df_matching['is_treatment'] == 1]
    df_control = df_matching[df_matching['is_treatment'] == 0]
    
    control_vecs = create_vecs(df_control, method="concat")
    treatment_vecs = create_vecs(df_treatment, method="concat")
    results_dir_knn = src_path / 'results' / 'knn'

    concat_matching_cols = treatment_vecs.columns.difference(['auth_id', 'auth_name', 'is_treatment']).to_list()
    
    match_and_plot(treatment_vecs, control_vecs, concat_matching_cols, k, metric, results_dir_knn)

def get_placebo_targets(results_dir_knn: str, treatment_ids: List[str], k: int, metric: str):
    # for each real target, create a placebo target
    # for each real target, read the results of k nearest neighbors, sample one of them and add to list
    
    placebo_targets = []
    placebo_names = []
    for target_id in treatment_ids:
        # read the results of k nearest neighbors
        results_filename = f'knn_results_k={k}_{metric}.csv'
        results_df = pd.read_csv(results_dir_knn / results_filename)
        target_results = results_df[(results_df['target_id'] == target_id) & (results_df['is_nn'])]
        # sample one of them
        sampled_target = target_results.sample(1)
        # sample again if the sampled target is already in the placebo_targets
        while sampled_target['control_id'].values[0] in placebo_targets:
            sampled_target = target_results.sample(1)
        placebo_targets.append(sampled_target['control_id'].values[0])
        placebo_names.append(sampled_target['control_name'].values[0])
    
    # save placebo targets to a file
    placebo_targets_df = pd.DataFrame(
        {'treatment_id': treatment_ids, 'placebo_id': placebo_targets, 'placebo_name': placebo_names})
    placebo_targets_df.to_csv(results_dir_knn / 'placebo_targets.csv', index=False)
    
    return placebo_targets

def run_matching_placebo(k, metric, placebo_targets):

    df_matching = pd.read_csv(src_path / 'data/matching_features_df.csv')
    df_matching = df_matching.rename(columns={'authority_code': 'auth_id', 'authority_name': 'auth_name'})
    
    all_dfs = []
    # for each placebo target get the k nearest neighbors
    for placebo_target in tqdm(placebo_targets):
        df_control = df_matching[(df_matching['is_treatment'] == 0)]
        control_vecs = create_vecs(df_control, method="concat")
        concat_matching_cols = control_vecs.columns.difference(['auth_id', 'auth_name', 'is_treatment']).to_list()
        # seperate placebo target from control
        placebo_target_vec = control_vecs[control_vecs['auth_id'] == placebo_target]
        placebo_control_vecs = control_vecs[control_vecs['auth_id'] != placebo_target]
        results = match_and_plot(placebo_target_vec, placebo_control_vecs, concat_matching_cols, k, metric, results_dir_knn, placebo=True)
        all_dfs.append(results)
        
    # Save the results DataFrame
    results_filename = f'knn_results_k={k}_{metric}_placebo=True.csv'
    all_targets_results = pd.concat(all_dfs, ignore_index=True)
    all_targets_results.to_csv(results_dir_knn / results_filename, index=False)

if __name__ == '__main__':
    src_path = Path.cwd() / 'src'
    results_dir_knn = src_path / 'results' / 'knn'
    
    # test_match_and_plot
    
    k = 10
    metric='cosine' # 'euclidean' or 'cosine' or 'manhattan' or 'minkowski' or 'chebyshev' or 'hamming' or 'jaccard'
    run_matching(k, metric)
    
    k_for_sampling_placebo = 3
    k = 10
    metric = 'cosine'
    placebo_targets = get_placebo_targets(results_dir_knn, treatment_ids, k=k_for_sampling_placebo, metric=metric)
    placebo_df = pd.read_csv(results_dir_knn / 'placebo_targets.csv')
    placebo_targets = placebo_df['placebo_id'].to_list()
    run_matching_placebo(k, metric, placebo_targets)
    