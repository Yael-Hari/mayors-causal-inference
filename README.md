# The Effect of Legal Actions Against Public Officials on Local Authority Metrics in Israel

This repository contains the code and data for the project titled "The Effect of Legal Actions Against Public Officials on Local Authority Metrics in Israel". The study examines the causal impact of legal actions against public officials—specifically arrests, indictments filed, and convictions of mayors or deputy mayors—on various socioeconomic and infrastructural metrics of local authorities in Israel from 1999 to 2022.

## Introduction
Corruption and misconduct among public officials can significantly affect governance, public trust, and community well-being. This project investigates how legal events involving public officials—such as arrests, indictments filed, and convictions—impact various metrics of local authorities in Israel, including migration trends, budget deficits, and infrastructure development.

We employ a K-Nearest Neighbors (KNN) matching approach combined with Difference-in-Differences (DiD) analysis to estimate the effects of these legal events.

## Data Sources
The primary data sources used in this project are:

- Local Authorities Data (1999-2022): Demographic, socioeconomic, infrastructural, and financial data for Israeli local authorities.
- Legal Events Data: Information on arrests, indictments filed, and convictions of mayors or deputy mayors, including the year and type of each event.

## Quick Start

**Set up the Python environment:**

    ```bash
    conda env create -f environment.yml
    conda activate legal_actions_env
    ```

### Main Analysis Steps

1. **Data Preprocessing and Feature Engineering:**

    - Run `src/data_preprocess.ipynb` to process raw data and imput data.
    - Output saved to `data/processed/inference_features_df.csv` or `data/processed/inference_features_df_imputed.csv`

2. **Matching Process:**

    - Run `src/matching.py` to perform KNN matching of treatment authorities with similar control authorities.
    - Outputs matching results and visualizations.
    - Results saved in `results/knn/`

3. **Difference-in-Differences Analysis:**

    - Run `src/diff_in_diff_funcs.py` to conduct the DiD analysis using four methods:
        - Ordinary Least Squares (OLS) Regression
        - Fixed Effects Model
        - Clustered Standard Errors
        - Permutation Tests
    - Results saved in `results/diff/`

4. **Analysis and Visualization:**

    - Run `src/plots.ipynb` to generate plots and figures.

## Contact Information

For any questions or issues, please contact:

- Keren Gruteke Klein: gkeren at campus.technion.ac.il
- Yael Hari: yael.hari at campus.technion.ac.il

## Acknowledgments
We would like to thank the Israeli Central Bureau of Statistics for providing access to the local authorities data. 
