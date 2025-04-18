import pandas as pd
import random
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import math
import streamlit as st # type: ignore
import folium
from folium.plugins import HeatMap
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

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
    parts = str(date_str).split('-')
    
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

def plot_distribution_charts(numeric_columns, dataset, country_name=""):
    """
    Plot distribution charts for numeric columns in a dataset.
    @param numeric_columns - List of numeric columns to plot distribution charts for.
    @param dataset - The dataset containing the numeric columns.
    @param country_name - Title for the distribution charts (default is an empty string).
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
        axes[i].set_title(f'Distribution of {col} for `{country_name}`')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

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
    # Step 1: Remove columns with too many missing values
    # Drop columns where more than 70% of values are missing (keep columns with at least 30% data)
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
    # missing_count = df.isnull().sum().sum()

    return df

def folium_map(data, target_column, zoom_start=4):
    """
    Create a folium map with a heatmap based on the given data and target column.
    @param data - The dataset containing latitude, longitude, and target column values.
    @param target_column - The column in the dataset to be used for the heatmap.
    @return A folium map with a heatmap based on the provided data.
    """
    # Ensure latitude and longitude columns are numeric and handle non-numeric values
    data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    data['longitude'] = pd.to_numeric(data['longitude'], errors='coerce')
    data[target_column] = pd.to_numeric(data[target_column], errors='coerce')

    # Drop rows with NaN values in latitude, longitude, or target_column
    data = data.dropna(subset=['latitude', 'longitude', target_column])

    # Check if the filtered dataset is empty
    if len(data) == 0:
        st.warning("No valid data points available for the map visualization.")
        # Return a default map centered on (0, 0)
        return folium.Map(location=[0, 0], zoom_start=2)

    centroid_lat = data['latitude'].mean()
    centroid_lon = data['longitude'].mean()
    map = folium.Map(location=[centroid_lat, centroid_lon], zoom_start=zoom_start)

    # Prepare data for the HeatMap plugin
    heat_data = [[row['latitude'], row['longitude'], row[target_column]] for index, row in data.iterrows()]

    # Add the HeatMap plugin to the map
    HeatMap(heat_data,
            min_opacity=0.05,
            max_opacity=0.9).add_to(map)


    return map

def calculate_SOC_stocks(p, BD, SOC, rf):
    """
    Calculate the Soil Organic Carbon (SOC) stocks based on the provided parameters.
    @param p - The fraction of horizon contributing to 30 cm depth
    @param BD - The bulk density in g/cm³
    @param SOC - The soil organic carbon concentration in g/kg
    @param rf - The rock fragment volume fraction (0-1)
    @return The SOC stock in t/ha
    """
    # Convert inputs to NumPy arrays for vectorized operations
    p = np.array(p)       # Fraction of each horizon contributing to 30 cm
    BD = float(BD)     # Bulk density (g/cm³)
    SOC = float(SOC)   # SOC concentration (g/kg)
    rf = float(rf)     # Rock fragment fraction (as decimal)

    # Compute SOC stocks using the given formula
    # Multiply by 1 to directly get t/ha (tons per hectare)
    SOC_stock = np.sum(p * BD * SOC * (1 - rf))
    # print the calculation for manual verification
    # print(f"{p} * {BD} * {SOC} * (1 - {rf}) = {SOC_stock}")

    return SOC_stock

def calculate_horizon_fractions(upper_depth, lower_depth, total_depth=30):
    """
    Calculate the fractions of the horizon depths relative to the total depth.
    @param upper_depth - The upper depth values.
    @param lower_depth - The lower depth values.
    @param total_depth - The total depth value (default is 30).
    @return The fractions of the horizon depths.
    """
    # Ensure the input arrays are numpy arrays for element-wise operations
    # upper_depth = np.array(upper_depth)
    # lower_depth = np.array(lower_depth)
    
    # Calculate the depth of each horizon
    horizon_depths = lower_depth - upper_depth
    
    # Calculate the fraction of each horizon in relation to the total depth
    fractions = horizon_depths / total_depth
    
    return fractions

def preprocess_categorical(df, categorical_features):
    # df[categorical_features] = df[categorical_features].astype(str)
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_cols = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_features))
    df = df.drop(columns=categorical_features)
    return pd.concat([df, encoded_df], axis=1), encoder

def calculate_confidence_score(model, X):
    """
    Calculate a confidence score for predictions based on the model type.
    
    Parameters:
        model: Trained model (must be from MODELS_LIST)
        X: Scaled input features (numpy array or DataFrame)
    
    Returns:
        Mean confidence score (higher = more confident)
    """
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        # Tree-based models: Measure variability across decision trees
        pred_leafs = model.apply(X)  # Get the decision path for each tree
        confidence_scores = np.std(pred_leafs, axis=1)  # Higher std = lower confidence
        return 1 / (1 + np.mean(confidence_scores))  # Normalize confidence (0 to 1)

    elif isinstance(model, (LinearRegression, SVR, KNeighborsRegressor, MLPRegressor)):
        # Non-tree models: Use prediction variance
        preds = model.predict(X)
        confidence_scores = np.std(preds)  # Higher std = lower confidence
        return 1 / (1 + confidence_scores)  # Normalize confidence (0 to 1)

    else:
        raise ValueError(f"Model type {type(model)} is not supported.")