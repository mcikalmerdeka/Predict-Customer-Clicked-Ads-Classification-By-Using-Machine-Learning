import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

# Import the preprocessed original data
url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/Clicked%20Ads%20Dataset.csv"
ori_df_preprocessed = pd.read_csv(url_ori_processed)
ori_df_preprocessed = ori_df_preprocessed.loc[:, ori_df_preprocessed.columns != "Clicked on Ad"]

# =====================================================================Functions for data pre-processing========================================================================

## Checking basic data information
def check_data_information(data, cols):
    list_item = []
    for col in cols:
        # Convert unique values to string representation
        unique_sample = ', '.join(map(str, data[col].unique()[:5]))
        
        list_item.append([
            col,                                           # The column name
            str(data[col].dtype),                          # The data type as string
            data[col].isna().sum(),                        # The count of null values
            round(100 * data[col].isna().sum() / len(data[col]), 2),  # The percentage of null values
            data.duplicated().sum(),                       # The count of duplicated rows
            data[col].nunique(),                           # The count of unique values
            unique_sample                                  # Sample of unique values as string
        ])

    desc_df = pd.DataFrame(
        data=list_item,
        columns=[
            'Feature',
            'Data Type',
            'Null Values',
            'Null Percentage',
            'Duplicated Values',
            'Unique Values',
            'Unique Sample'
        ]
    )
    return desc_df







