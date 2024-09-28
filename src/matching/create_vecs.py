from typing import Union, Literal
import pandas as pd

key_cols = ['auth_id', 'auth_name', 'is_treatment']

fixed_features_cols = ['district', 'distance_from_tel_aviv', 'year_of_municipal_status']

temporal_features_cols = ['council_members_number', 'socioeconomic_level', 'compactness_cluster',
    'peripheral_cluster', 'total_jurisdiction_area', 'population_density',
    'total_population', 'jews_and_others', 'total_men', 'total_women',
    'dependency_ratio', 'natural_increase_per_1000',
    'total_migration_balance', 'immigrants_post_1990',
    'population_growth_rate', 'residential_percent_area',
    'education_percent_area', 'health_welfare_percent_area',
    'public_services_percent_area', 'culture_leisure_sports_percent_area',
    'commerce_percent_area', 'industry_infrastructure_percent_area',
    'transportation_percent_area', 'agricultural_buildings_percent_area',
    'public_open_space_percent_area', 'forest_woodland_percent_area',
    'orchards_citrus_olive_percent_area', 'cultivated_fields_percent_area',
    'other_open_space_percent_area', 'total_classes', 'total_students',
    'avg_students_per_class', 'percent_bagrut_eligible',
    'percent_university_bagrut', 'water_receipts', 'total_private_cars',
    'avg_car_age', 'accidents_per_1000_residents',
    'accidents_per_1000_vehicles', 'unemployment_benefits_total',
    'male_unemployment_benefits', 'avg_age_unemployment_recipients',
    'income_support_recipients_yearly', 'avg_salary_employees',
    'avg_salary_male', 'avg_salary_female', 'change_salary_emp_yoy',
    'change_salary_male_yoy', 'change_salary_female_yoy', 'employees_total',
    'percent_earning_min_wage', 'self_employed_total',
    'avg_income_self_employed', 'change_income_self_employed_yoy',
    'percent_earning_half_avg_salary', 'inequality_index_gini',
    'total_revenues', 'revenues_change', 'regular_budget_revenues',
    'gov_participation', 'gov_participation_change',
    'special_budget_revenues', 'property_tax_total_prev_year',
    'property_tax_residential_prev_year', 'property_tax_business_prev_year',
    'property_tax_total_current_year',
    'property_tax_residential_current_year',
    'property_tax_business_current_year']

def create_vecs(df_matching: pd.DataFrame, method: Union[Literal['agg'], Literal['concat']]) -> pd.DataFrame:
    agg_df_list = []
    concat_df_list = []

    # Loop through each authority
    for auth_id, group in df_matching.groupby('auth_id'):
        # Extract key columns for this authority (assuming they are the same across all years for an authority)
        key_values = group[key_cols].iloc[0]

        # Extract fixed feature values for this authority (assuming they are the same across all years)
        fixed_features = group[fixed_features_cols].iloc[0]

        if method == 'agg':
            # Aggregate temporal feature values over the years for 'agg' method
            agg_temporal_features = group[temporal_features_cols].mean().add_prefix('agg_')
            # Combine key values, fixed features, and aggregated temporal features
            agg_row = pd.concat([key_values, fixed_features, agg_temporal_features])
            agg_df_list.append(agg_row)

        elif method == 'concat':
            # Concatenate temporal feature values over the years for 'concat' method
            concat_temporal_features = group[temporal_features_cols].values.flatten()
            concat_feature_names = [f'{col}_{year}' for year in group['year'] for col in temporal_features_cols]
            
            # Create a Series with concatenated temporal features and add fixed features and key values
            concat_row = pd.Series(concat_temporal_features, index=concat_feature_names)
            concat_row = pd.concat([key_values, fixed_features, concat_row])
            concat_df_list.append(concat_row)

    # Create final DataFrame based on the selected method
    if method == 'agg':
        return pd.DataFrame(agg_df_list)
    elif method == 'concat':
        return pd.DataFrame(concat_df_list)
    
    
if __name__ == '__main__':
    # df_matching = pd.read_csv('matching_data.csv')
    # agg_vecs = create_vecs(df_matching, 'agg')
    # concat_vecs = create_vecs(df_matching, 'concat')
    # print(agg_vecs.head())
    # print(concat_vecs.head())
    
    # Sample data for testing
    df_matching = pd.DataFrame({
        'auth_id': [1, 1, 1, 2, 2],
        'auth_name': ['Authority A', 'Authority A', 'Authority A', 'Authority B', 'Authority B'],
        'is_treatment': [0, 0, 0, 1, 1],
        'district': [3, 3, 3, 5, 5],
        'distance_from_tel_aviv': [100.5, 100.5, 100.5, 200.0, 200.0],
        'year_of_municipal_status': [1950, 1950, 1950, 1960, 1960],
        'year': [2001, 2002, 2003, 2002, 2001],
        'council_members_number': [10, 15, 20, 30, 35],
        'socioeconomic_level': [5, 6, 7, 8, 9],
        'compactness_cluster': [1, 2, 3, 4, 5]
    })

    # Defining key columns, fixed features, and temporal features
    key_cols = ['auth_id', 'auth_name', 'is_treatment']
    fixed_features_cols = ['district', 'distance_from_tel_aviv', 'year_of_municipal_status']
    temporal_features_cols = ['council_members_number', 'socioeconomic_level', 'compactness_cluster']
    agg_vecs = create_vecs(df_matching, 'agg')
    concat_vecs = create_vecs(df_matching, 'concat')
