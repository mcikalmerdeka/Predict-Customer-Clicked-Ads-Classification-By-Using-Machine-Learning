import streamlit as st
import os
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import all preprocessing functions
from preprocessing import (check_data_information)

from feature_definitions import get_feature_definitions

# Page Config
st.set_page_config(page_title="Ad Click Prediction Dashboard", layout="wide")
st.title("Digital Ad Performance Analyzer")

# Author Information
st.markdown("""
#### Author
Developed by : Muhammad Cikal Merdeka | Data Analyst/Data Scientist

- [GitHub Profile](https://github.com/mcikalmerdeka)  
- [LinkedIn Profile](https://www.linkedin.com/in/mcikalmerdeka)
""")

# Add information about the app
with st.expander("**Read Instructions First: About This App**"):
    st.markdown("""
    ## Digital Ad Performance Analyzer

    ### 📌 Problem Statement
    A digital marketing company in Indonesia faces challenges in determining the effectiveness of their advertisements and identifying the right target audience. The current approach lacks precision in reaching potential customers who are likely to engage with the ads, resulting in inefficient ad spending and lower click-through rates. This leads to reduced return on investment (ROI) on their advertising campaigns and potential loss of market opportunities.
    
    ### 🎯 Goals & Objectives
    - **Primary Goal**: Enhance ad targeting by implementing machine learning to identify potential customers most likely to click on ads
    - **Secondary Goal**: Increase ROI by optimizing advertising spend for the most receptive audience segments
    
    The machine learning model aims to:
    - Accurately predict which users are most likely to click on advertisements
    - Identify patterns in customer behavior to optimize targeting strategies
    - Reduce advertising costs while improving engagement rates

    ### 📊 Key Business Metrics
    Two critical metrics that demonstrate our project's success:

    #### Primary Metric: Click-Through Rate (CTR)
    - Measures the percentage of users who click on the advertisement
    - Calculated as: (Number of Clicks / Total Ad Views) × 100
    - Improved from 50% to 99.8% after model implementation
    - Indicates effectiveness of targeting strategy

    #### Secondary Metric: Return on Ad Spend (ROAS)
    - Measures revenue generated per rupiah spent on advertising
    - Improved from 1.25 to 2.43 IDR
    - Demonstrates financial efficiency of ad campaigns
    - Profit increased from Rp.1,500,000 to Rp.8,727,810 (581.8% increase)

    ### 🔍 **How to Use the App**
                
    #### Prediction Options:
    - Input new customer data for ad targeting analysis
    - Receive instant prediction of click probability

    #### Data Input Methods:
    - A. Individual Customer Analysis
        - Enter details for a single customer through the form
    - B. Batch Data Processing
        - Upload multiple customer records
        - Ensure dataset includes: daily site time, age, area income, internet usage, etc.
    - Note: Sample data is available for testing the model

    #### Data Processing Pipeline:
    The application processes customer data through these steps:
    1. Data Type Conversion
        - Standardize input formats
    2. Missing Value Treatment
        - Handle incomplete data points
    3. Outlier Management
        - Address extreme values
    4. Feature Engineering
        - Create derived insights from raw data
    5. Feature Encoding
        - Convert categorical data to numerical format
    6. Feature Selection
        - Focus on most predictive variables
    7. Data Scaling
        - Normalize data for consistent analysis

    ### 🤖 Model Performance
    - Implements a tuned Logistic Regression model
    - Accuracy: 97.3%
    - Significant improvements in both CTR and ROAS
    - Trained on comprehensive customer behavior dataset

    ### ⚠️ <span style="color:red;"> Important Notes </span>
    - Predictions are probability-based and should guide, not replace, marketing strategy
    - **Combine model insights with marketing expertise for best results**
    - Regular model updates recommended to maintain performance
    - Results may vary based on market conditions and campaign specifics
    """, unsafe_allow_html=True)


# Load pre-trained model
@st.cache_resource
def load_model():
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    model_path = os.path.join(parent_dir, 'models', 'tuned_logistic_regression_model.joblib')
    
    return joblib.load(model_path)

model = load_model()