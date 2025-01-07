import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.stats as stats

# Import the preprocessed original data
url_ori_processed = "https://raw.githubusercontent.com/mcikalmerdeka/Predict-Customer-Clicked-Ads-Classification-By-Using-Machine-Learning/main/df_model.csv"
ori_df_preprocessed = pd.read_csv(url_ori_processed, index_col=0)
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

## Initial data transformation
def initial_data_transform(data):

    # Rename column name for and maintain column name similarity
    data = data.rename(columns={'Male': 'Gender',
                                'Timestamp': 'Visit Time',
                                'city' : 'City',
                                'province' : 'Province',
                                'category' : 'Category'})

    # Re-arrange column (target 'Clicked on Ad' at the end --> personal preference)
    data = data[[col for col in data.columns if col != 'Clicked on Ad'] + ['Clicked on Ad']]

    # Change data type of Visit Time to datetime
    data['Visit Time'] = pd.to_datetime(data['Visit Time'])

    return data





