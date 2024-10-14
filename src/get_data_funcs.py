import pandas as pd
from pathlib import Path

def get_infer_df(imupted: bool):
    """
    Load the inference DataFrame and return it.
    
    Parameters:
    imupted (bool): Whether to load the imputed inference DataFrame.
    
    Returns:
    pd.DataFrame: The inference DataFrame.
    """
    src_path = Path.cwd() / 'src'
    if imupted:
        infer_df = pd.read_csv(src_path / 'data/inference_features_df_imputed.csv')
    else: 
        infer_df = pd.read_csv(src_path / 'data/inference_features_df.csv')
    infer_df = infer_df.rename(columns={'authority_code': 'auth_id'})
    id_name_df = pd.read_csv(src_path / 'data/authority_id_name.csv')
    infer_df = infer_df.merge(id_name_df, on='auth_id', how='left')
    return infer_df

def get_placebo_incident_df():
    # TODO: Implement this function
    pass