percentage_cols = ['immigrants_post_1990']
numeric_cols = [
    'distance_from_tel_aviv', 
    'council_members_number', 
    'socioeconomic_level',
    'compactness_cluster', 
    'peripheral_cluster'
]
only_one_year_needed_col = ['year_of_municipal_status']

columns_to_predict = ['cumulative_deficit']
treatment_ids = [1034, 2630, 2650, 3000, 6200, 6500, 6700, 6900, 7000, 7100, 8000, 8400, 8600, 9100]
placebo_ids = [] # TODO: Add placebo IDs

# columns_to_predict = [
#     'total_in_migration', 'male_in_migration', 'female_in_migration',
#     'in_migration_age_15_29', 'total_out_migration', 'male_out_migration',
#     'female_out_migration', 'out_migration_age_15_29',
#     'start_roads_length_km', 'start_roads_area_sqm', 'start_water_pipes_km',
#     'start_sewage_pipes_km', 'start_drainage_pipes_km',
#     'complete_roads_length_km', 'complete_roads_area_sqm',
#     'complete_water_pipes_km', 'complete_sewage_pipes_km',
#     'complete_drainage_pipes_km', 'satisfaction_municipal_services',
#     'satisfaction_municipal_performance', 'satisfaction_residential_area',
#     'satisfaction_cleanliness', 'satisfaction_parks_green_spaces',
#     'feeling_safe_dark', 'contacting_local_authority', 'total_expenditures',
#     'expenditures_change', 'operations_expenditures',
#     'loan_repayment_expenditures', 'financing_expenditures',
#     'special_budget_expenditures', 'budget_surplus_deficit',
#     'annual_surplus_deficit', 'cumulative_deficit'
# ]