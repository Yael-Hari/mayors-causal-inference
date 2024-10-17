import pandas as pd
from pathlib import Path
from src.const import treatment_ids

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

def get_placebo_infer_df(imupted: bool, src_path):
    results_dir_knn = src_path / 'results' / 'knn'
    placebo_targets_df = pd.read_csv(results_dir_knn / 'placebo_targets.csv')
    infer_df = get_infer_df(imupted=imupted)
    new_infer_df = create_new_df_by_placebo_ids(infer_df, placebo_targets_df, treatment_ids)
    return new_infer_df

def create_new_df_by_placebo_ids(infer_df, placebo_targets_df, treatment_ids)->pd.DataFrame:
    """
    Create a placebo DataFrame by replacing the treatment incidents with placebo incidents.

    Args:
        imupted (bool): Whether to load the imputed inference DataFrame.

    Returns:
        pd.DataFrame: The new Inference DataFrame with placebo incidents.
    """
    for target_id in treatment_ids:
        # get target incident types and incident years from infer_df
        target_incidents = infer_df[(infer_df['auth_id'] == target_id)][['incident_year', 'incident_type']].drop_duplicates()
        # get placebo target from placebo_targets_df
        placebo_target = placebo_targets_df[placebo_targets_df['treatment_id'] == target_id]['placebo_id'].values[0]
        # get placebo rows from infer_df
        placebo_rows = infer_df[(infer_df['auth_id'] == placebo_target)]
        # drop original rows of placebo target from infer_df
        infer_df = infer_df[~(infer_df['auth_id'] == placebo_target)]
        
        # replace placebo rows with new rows with incident type and year
        for _, row in target_incidents.iterrows():
            incident_year = row['incident_year']
            incident_type = row['incident_type']
            new_placebo_target_incidents = placebo_rows.copy()
            new_placebo_target_incidents['incident_year'] = incident_year
            new_placebo_target_incidents['incident_type'] = incident_type
            infer_df = pd.concat([infer_df, new_placebo_target_incidents])
            
    infer_df['incident_year'] = infer_df['incident_year'].astype("int64")
    infer_df.reset_index(drop=True, inplace=True)
    return infer_df
    

def test_create_new_df_by_placebo_ids():
    infer_df = pd.DataFrame({
        'auth_id': [1, 1, 2, 2, 2, 2, 3, 3, 4, 4],
        'incident_year': [2000, 2000, 2001, 2001, 2001, 2001, None, None, None, None],
        'incident_type': ['A', 'A', 'A', 'A', 'B', 'B', None, None, None, None],
        'value': [1, 2, 3, 4, 3, 4, 5, 6, 7, 8]
    })
    placebo_targets_df = pd.DataFrame({
        'treatment_id': [1, 2],
        'placebo_id': [3, 4]
    })
    treatment_ids = [1, 2]
    expected_df = pd.DataFrame({
        'auth_id': [1, 1, 2, 2, 2, 2, 3, 3, 4, 4, 4, 4],
        'incident_year': [2000, 2000, 2001, 2001, 2001, 2001, 2000, 2000, 2001, 2001, 2001, 2001],
        'incident_type': ['A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'A', 'A', 'B', 'B',],
        'value': [1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 7, 8]
    })
    new_infer_df = create_new_df_by_placebo_ids(infer_df, placebo_targets_df, treatment_ids)
    pd.testing.assert_frame_equal(new_infer_df, expected_df)

if __name__ == '__main__':
    test_create_new_df_by_placebo_ids()
    
    # print placebo_ids
    src_path = Path.cwd() / 'src'
    results_dir_knn = src_path / 'results' / 'knn'
    placebo_targets_df = pd.read_csv(results_dir_knn / 'placebo_targets.csv')
    print("placebo_ids: ", placebo_targets_df['placebo_id'].unique().tolist())