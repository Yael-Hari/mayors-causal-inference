import pandas as pd
from pathlib import Path
from src.diff_in_diff.diff_in_diff_funcs import calc_diff_in_diff

src_path = Path.cwd() / 'src'
results_dir = src_path / 'results'
diff_results_dir = results_dir / 'diff'

df_matching = pd.read_csv(src_path / 'data/matching_features_df.csv')
df_matching = df_matching.rename(columns={'authority_code': 'auth_id', 'authority_name': 'auth_name'})
df_treatment = df_matching[df_matching['is_treatment'] == 1]
df_control = df_matching[df_matching['is_treatment'] == 0]

infer_df = pd.read_csv(src_path / 'data/inference_features_df.csv')
infer_df = infer_df.rename(columns={'authority_code': 'auth_id'})
infer_df = infer_df.merge(df_matching[['auth_id', 'auth_name']], on='auth_id', how='left')

columns_to_predict = [
    'total_in_migration', 'male_in_migration', 'female_in_migration',
    'in_migration_age_15_29', 'total_out_migration', 'male_out_migration',
    'female_out_migration', 'out_migration_age_15_29',
    'start_roads_length_km', 'start_roads_area_sqm', 'start_water_pipes_km',
    'start_sewage_pipes_km', 'start_drainage_pipes_km',
    'complete_roads_length_km', 'complete_roads_area_sqm',
    'complete_water_pipes_km', 'complete_sewage_pipes_km',
    'complete_drainage_pipes_km', 'satisfaction_municipal_services',
    'satisfaction_municipal_performance', 'satisfaction_residential_area',
    'satisfaction_cleanliness', 'satisfaction_parks_green_spaces',
    'feeling_safe_dark', 'contacting_local_authority', 'total_expenditures',
    'expenditures_change', 'operations_expenditures',
    'loan_repayment_expenditures', 'financing_expenditures',
    'special_budget_expenditures', 'budget_surplus_deficit',
    'annual_surplus_deficit', 'cumulative_deficit'
]

infer_df.set_index('auth_id', inplace=True)

calc_diff_in_diff(
    infer_df,
    columns_to_predict=columns_to_predict, 
    treatment_ids=df_treatment['auth_id'].unique(),
    k=10,
    distance_metric='cosine',
    results_dir=results_dir,
    diff_results_dir=diff_results_dir,
)