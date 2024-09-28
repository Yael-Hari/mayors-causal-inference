import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def match_and_plot(df, matching_cols, k, target_auth_code, metric='euclidean'):
    """
    Find the k nearest neighbors for a specific authority, plot the results with dimensionality reduction,
    and save both the plot and the results DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    matching_cols (list): The columns to be used for matching.
    k (int): The number of nearest neighbors to find.
    target_auth_code: The identifier of the target authority for which to find neighbors.
    metric (str): The distance metric to be used in NearestNeighbors.
    
    Returns:
    pd.DataFrame: A DataFrame containing the k nearest neighbors for the target authority.
    """

    # Check for columns with null values and print them
    null_columns = df[matching_cols].columns[df[matching_cols].isnull().any()]
    if len(null_columns) > 0:
        print(f"Columns with null values: {list(null_columns)}")
        return None

    # Ensure the target authority exists in the DataFrame
    if target_auth_code not in df.index:
        print(f"Target authority '{target_auth_code}' not found in the DataFrame.")
        return None
    
    # Fit the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric=metric)
    nn.fit(df[matching_cols])
    
    # Find the nearest neighbors for the target authority
    target_index = df.index.get_loc(target_auth_code)
    distances, indices = nn.kneighbors(df[matching_cols].iloc[[target_index]])
    
    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[matching_cols])
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot all points in grey
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], color='grey', label='Others')

    # Highlight the k nearest neighbors in yellow
    nn_indices = indices[0][1:]  # Skip the first index, which is the target itself
    plt.scatter(reduced_data[nn_indices, 0], reduced_data[nn_indices, 1], color='yellow', label=f'Top {k} Neighbors')

    # Highlight the target authority in green
    plt.scatter(reduced_data[target_index, 0], reduced_data[target_index, 1], color='green', label='Target Authority')
    
    # Add labels for all points
    for i in range(len(reduced_data)):
        plt.text(reduced_data[i, 0] + 0.01, reduced_data[i, 1] + 0.01, df.iloc[i]['auth_name'], fontsize=9)
    
    # Add legend
    plt.legend()

    # Save the plot
    plot_filename = f'knn_plot_{target_auth_code}_{metric}.png'
    plt.savefig(plot_filename)
    plt.close()

    print(f"Plot saved as {plot_filename}")

    # Prepare the results DataFrame
    results = []
    for i in range(len(df)):
        results.append({
            'auth_code': df.index[i],
            'auth_name': df.iloc[i]['auth_name'],
            'target_auth': target_auth_code,
            'k': k,
            'is_nn': df.index[i] in df.index[nn_indices]
        })

    results_df = pd.DataFrame(results)
    
    # Save the results DataFrame
    df_filename = f'knn_results_{target_auth_code}_{metric}.csv'
    results_df.to_csv(df_filename, index=False)
    
    print(f"Results DataFrame saved as {df_filename}")

    return results_df

if __name__ == '__main__':
    metric = 'euclidean' # 'euclidean' or 'cosine' or 'manhattan' or 'minkowski' or 'chebyshev' or 'hamming' or 'jaccard'

    # Example DataFrame
    df = pd.DataFrame({
        'auth_code': [1, 2, 3, 4, 5],
        'auth_name': ['A', 'B', 'C', 'D', 'E'],
        'feature1': [0.1, 0.3, 0.2, 0.5, 0.4],
        'feature2': [1.2, 0.9, 1.5, 1.0, 1.3]
    })
    df.set_index('auth_code', inplace=True)

    # Call the function
    results_df = match_and_plot(df, matching_cols=['feature1', 'feature2'], k=2, target_auth_code=2, metric='euclidean')