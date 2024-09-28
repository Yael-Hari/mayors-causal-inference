import pandas as pd
from src.const import DATASET_MATCHING_PATH

df_matching = pd.read_csv(DATASET_MATCHING_PATH, index_col=False)
df_treatment = df_matching[df_matching['is_treatment'] == 1]
df_control = df_matching[df_matching['is_treatment'] == 0]

