
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_across_hyper_params(compare_df_path: str) -> None:
    """
    Analyze and plot the effects of hyperparameters (distance metric and k) on the results.
    
    1. Effect of distance metric on the results:
        - X-axis: distance metric
        - Y-axis: mean_DiD, median_DiD, min_DiD, max_DiD (manual box plot)
        - Plot for k=3 and k=5 separately
        - Create subplots for each metric
    2. Effect of k on the results:
        - X-axis: k
        - Y-axis: mean_DiD, median_DiD, min_DiD, max_DiD (manual box plot)
        - Plot for distance metric "cosine"
        - Create subplots for each metric
    """
    
    # Load the compare_df
    compare_df = pd.read_csv(compare_df_path)
    
    # Set up the plotting area for distance metric analysis
    metrics = compare_df['metric'].unique()
    ks = [3, 5]

    for k in ks:
        plt.figure(figsize=(16, 8))
        for i, metric in enumerate(metrics):
            plt.subplot(1, len(metrics), i+1)
            subset = compare_df[(compare_df['k'] == k) & (compare_df['metric'] == metric)]
            sns.boxplot(data=subset, x='distance_metric', y='mean_DiD', color='blue', label='mean_DiD')
            sns.boxplot(data=subset, x='distance_metric', y='median_DiD', color='green', label='median_DiD')
            sns.boxplot(data=subset, x='distance_metric', y='min_DiD', color='red', label='min_DiD')
            sns.boxplot(data=subset, x='distance_metric', y='max_DiD', color='orange', label='max_DiD')
            plt.title(f'Distance Metric Effect on {metric} (k={k})')
            plt.ylabel('DiD')
            plt.xlabel('Distance Metric')
            plt.xticks(rotation=45)
            if i == len(metrics) - 1:
                plt.legend()
        plt.tight_layout()
        plt.savefig(f'distance_metric_effect_k{k}.png')
        plt.close()
        print(f"Distance metric effect plot saved for k={k} as distance_metric_effect_k{k}.png")

    # Set up the plotting area for k analysis with distance metric "cosine"
    plt.figure(figsize=(16, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(1, len(metrics), i+1)
        subset = compare_df[(compare_df['distance_metric'] == 'cosine') & (compare_df['metric'] == metric)]
        sns.boxplot(data=subset, x='k', y='mean_DiD', color='blue', label='mean_DiD')
        sns.boxplot(data=subset, x='k', y='median_DiD', color='green', label='median_DiD')
        sns.boxplot(data=subset, x='k', y='min_DiD', color='red', label='min_DiD')
        sns.boxplot(data=subset, x='k', y='max_DiD', color='orange', label='max_DiD')
        plt.title(f'Effect of k on {metric} (Distance Metric: Cosine)')
        plt.ylabel('DiD')
        plt.xlabel('k')
        if i == len(metrics) - 1:
            plt.legend()
    plt.tight_layout()
    plt.savefig('src/plots/k_effect_cosine.png')
    plt.close()
    print("Effect of k plot saved for cosine distance metric as k_effect_cosine.png")

if __name__ == '__main__':
    # Call the function to generate and save plots
    plot_across_hyper_params(compare_df_path='compare_df.csv')