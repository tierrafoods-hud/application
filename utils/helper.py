import pandas as pd
import random
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st

def filter_by_country(df: pd.DataFrame, country_name: str) -> Optional[pd.DataFrame]:
    """

    Filter a DataFrame by a specific country name.
    @param df - The pandas DataFrame to filter.
    @param country_name - The name of the country to filter by.
    @return A filtered DataFrame containing only the rows with the specified country name, or None if the country name is not found in the DataFrame.
    """
    return df[df['country_name'] == country_name] if country_name in df['country_name'].unique() else None

def replace_invalid_dates(date_str, default_start_date=pd.Timestamp('1900-01-01'), default_end_date=pd.Timestamp('2024-12-31')):
    """
    Replace invalid dates in a date string with random valid dates within a specified range.
    @param date_str - The input date string to be checked and replaced if invalid.
    @return A Pandas datetime object with the replaced valid date.
    """
    # Split the date string into components
    parts = date_str.split('-')
    
    # Initialize default values for year, month, and day
    year, month, day = random.randint(default_start_date.year, default_end_date.year), random.randint(1, 12), random.randint(1, 28) # ensuring no wrong values for feb

    # Check and validate each part of the date
    if len(parts) >= 1 and parts[0].isdigit() and len(parts[0]) == 4:
        year = parts[0]
    if len(parts) >= 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 12:
        month = parts[1].zfill(2)
    if len(parts) >= 3 and parts[2].isdigit() and 1 <= int(parts[2]) <= 31:
        day = parts[2].zfill(2)
    
    # Construct the valid date string
    valid_date_str = f"{year}-{month}-{day}"
    
    # Convert to datetime
    return pd.to_datetime(valid_date_str, errors='coerce')

def plot_distribution_charts(numeric_columns, dataset, title=""):
    """
    Plot distribution charts for numeric columns in a dataset.
    @param numeric_columns - List of numeric columns to plot distribution charts for.
    @param dataset - The dataset containing the numeric columns.
    @param title - Title for the distribution charts (default is an empty string).
    @return None
    """
    num_cols = len(numeric_columns)

    # Define the number of rows and columns dynamically
    chart_cols = 3  # Fixed number of columns
    chart_rows = math.ceil(num_cols / chart_cols)  # Calculate required rows based on columns

    fig, axes = plt.subplots(chart_rows, chart_cols, figsize=(15, 3 * chart_rows), constrained_layout=True)
    axes = axes.flatten()  # Flatten the axes for easy iteration

    for i, col in enumerate(numeric_columns):
        sns.histplot(dataset[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Column distribution {title}")
    plt.show()

    return fig


@st.cache_data(ttl=600)
def remove_outliers(df):
    """
    Remove outliers from a DataFrame based on the z-score method.
    @param df - The DataFrame containing numerical columns.
    @return DataFrame with outliers removed.
    """
    numeric_columns_dataset = df.select_dtypes(include=['number']).columns
    for col in numeric_columns_dataset:
        # setting upper and lower limit
        upper_limit = df[col].mean() + 3*df[col].std()
        lower_limit = df[col].mean() - 3*df[col].std()

        # count outliers
        count_outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)]

        # trim outlier using z-score
        df = df[(df[col] >= lower_limit) & (df[col] <= upper_limit)]
    
    return df

@st.cache_data(ttl=600)
def preprocess_data(df):
    """
    Preprocess the input DataFrame by performing the following steps:
    1. Remove columns with more than 30% missing values.
    2. Remove rows with negative values.
    3. Remove outliers using the z-score method.
    4. Fill any remaining missing values with column means.
    @param df - The input DataFrame to be preprocessed.
    @return The preprocessed DataFrame.
    """
    with st.spinner('Preprocessing data...'):
        # Step 1: Remove columns with too many missing values
        df = df.dropna(thresh=0.3 * len(df), axis=1)
        
        # Step 2: Remove negative values
        # df = df[df.select_dtypes(include=['number']).ge(0).all(axis=1)] # removed all the
        
        # Step 3: Remove outliers
        df = remove_outliers(df)
        
        # Step 4: Handle remaining missing values
        # Get numeric columns excluding lat/lon
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = numeric_cols.drop(['latitude', 'longitude'], errors='ignore')
        
        # Calculate means once for all numeric columns
        col_means = df[numeric_cols].mean()
        
        # Fill missing values in numeric columns efficiently
        df[numeric_cols] = df[numeric_cols].fillna(col_means)

        # Handle silt and clay columns if present
        if {'silt', 'clay'}.issubset(df.columns):
            df['silt_plus_clay'] = df[['silt', 'clay']].sum(axis=1, skipna=True)
            df.drop(columns=['silt', 'clay'], inplace=True)

        # Calculate organic matter metrics if orgc present
        if 'orgc' in df.columns:
            # Use vectorized operations for better performance
            organic_matter = 1.724 * df['orgc']
            df = df.assign(
                organic_matter=organic_matter,
                bulk_density=1.62 - 0.06 * organic_matter
            )

        # Get final missing value count
        missing_count = df.isnull().sum().sum()

    return df
