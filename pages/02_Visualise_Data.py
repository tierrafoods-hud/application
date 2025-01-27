import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from typing import List, Optional
import logging
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from utils.display_table import display_table_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cache data loading
@st.cache_data
def load_datasets(datasets: List[str]) -> pd.DataFrame:
    """Load and concatenate multiple datasets"""
    load_data = [pd.read_csv(dataset) for dataset in datasets]
    return pd.concat(load_data, ignore_index=True)

@st.cache_data
def validate_config(dataset, map_file):
    """Validate file existence"""
    # for dataset in datasets:
    # if not os.path.exists(dataset):
    #     st.error(f"Dataset file not found: {dataset}")
    #     return False
    if not os.path.exists(map_file):
        st.error(f"Map file not found: {map_file}")
        return False
    return True

# Main app
st.title("Soil Data Visualization")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Dataset Configuration", expanded=True):
        datasets = st.file_uploader(
            "Upload Dataset",
            type=['csv'],
            help="Upload one or more CSV files containing soil data"
        )
        # datasets = [file.name for file in uploaded_files] if uploaded_files else []
        
        dataset_title = st.text_input("Dataset Title", "Mexico")
        map_file = st.text_input("Map File Path", "./data/mexico.geojson")
        
        threshold_of_missing_values = st.slider(
            "Missing Values Threshold",
            0.0, 1.0, 0.3
        )

    with st.expander("Column Configuration"):
        target_column = st.selectbox(
            "Target Column",
            ["orgc_value", "silt_value", "clay_value"]
        )
        
        orgc_column_name = st.text_input("Organic Carbon Column", "orgc_value", help="The column name to calculate organic matter and bulk density")
        silt_column_name = st.text_input("Silt Column", "silt_value", help="The column name to calculate organic matter and bulk density")
        clay_column_name = st.text_input("Clay Column", "clay_value", help="The column name to calculate organic matter and bulk density")
    
    with st.expander("Spatial Configuration"):
            col1, col2 = st.columns(2)
            with col1:
                west_lon = st.number_input("Western Longitude", value=-117.12)
                south_lat = st.number_input("Southern Latitude", value=14.53)
            with col2:
                east_lon = st.number_input("Eastern Longitude", value=-86.81)
                north_lat = st.number_input("Northern Latitude", value=32.72)
            
            spatial_bounding_coordinates = [west_lon, south_lat, east_lon, north_lat]

# Main content
if validate_config(datasets, map_file):
    
    master_dataset = None
    if not datasets:
        st.error("Please upload a dataset")
        exit()
    else:
        with st.spinner("Loading data..."):
            master_dataset = pd.read_csv(datasets)

    tab1, tab2, tab3 = st.tabs(["Data Overview", "Temporal Analysis", "Spatial Analysis"])
    
    with tab1:
        st.subheader("Data Preview")
        
        display_table_data(master_dataset)
        
        # Missing values heatmap
        st.subheader("Missing Values Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(master_dataset.isnull(), cbar=False, cmap='viridis')
        plt.title(f'Missing values in the {dataset_title} dataset')
        st.pyplot(fig)
        
        # Distribution plots
        st.subheader("Distribution Analysis")
        numeric_columns = master_dataset.select_dtypes(include=['float64', 'int64']).columns
        
        @st.cache_data
        def plot_distributions(data, _columns):
            chart_cols = 3
            chart_rows = -(-len(_columns) // chart_cols)
            fig, axes = plt.subplots(chart_rows, chart_cols, 
                                figsize=(15, 3 * chart_rows), 
                                constrained_layout=True)
            axes = axes.flatten()
            
            for i, col in enumerate(_columns):
                sns.histplot(data[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
                
            return fig
            
        st.pyplot(plot_distributions(master_dataset, numeric_columns))

    with tab2:
        with st.expander("Temporal Configuration", expanded=True):
            temporal_column_name = st.text_input("Time Column", "date")
            temporal_feature_columns = st.multiselect(
                "Features to Analyze",
                master_dataset.columns.tolist(),
                ["orgc_value", "silt_value", "clay_value"]
            )

        col1, col2 = st.columns(2)
        with col1:
            temporal_bounding_years_from = st.number_input(
                "Year From",
                min_value=1900,
                max_value=datetime.now().year,
                value=2005,
                key="year_from"  # Add unique key
            )
        with col2:
            temporal_bounding_years_to = st.number_input(
                "Year To",
                min_value=1900,
                max_value=datetime.now().year,
                value=2005,
                key="year_to"  # Add unique key
            )

        st.subheader("Temporal Analysis")
        
        @st.cache_data(show_spinner=False)
        def prepare_temporal_data(dataset, year_from, year_to):  # Add year parameters
            dataset = dataset.copy()
            dataset['organic_matter'] = dataset[orgc_column_name] * 1.724
            dataset['bulk_density'] = 1.62 - 0.06 * dataset['organic_matter']
            dataset['silt_plus_clay'] = (
                dataset[silt_column_name].fillna(0) + 
                dataset[clay_column_name].fillna(0)
            )

            # filter by year
            dataset = dataset[
                (dataset[temporal_column_name].dt.year >= year_from) & 
                (dataset[temporal_column_name].dt.year <= year_to)
            ]

            # filter by coordinates
            dataset = dataset[
                (dataset['latitude'] >= south_lat) & 
                (dataset['latitude'] <= north_lat) &
                (dataset['longitude'] >= west_lon) &
                (dataset['longitude'] <= east_lon)
            ]

            return dataset

        # set datetime column
        master_dataset[temporal_column_name] = pd.to_datetime(master_dataset[temporal_column_name])

        temporal_data = prepare_temporal_data(
            master_dataset,
            temporal_bounding_years_from,
            temporal_bounding_years_to
        )
        
        # Plot temporal features
        for feature in temporal_feature_columns:
            fig, ax = plt.subplots(figsize=(12, 4))
            sns.lineplot(data=temporal_data,
                        x=temporal_column_name,
                        y=feature, ax=ax)
            ax.set_title(f'Time series of {feature}')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with tab3:
        st.subheader("Spatial Analysis")
        
        @st.cache_data
        def prepare_spatial_data(dataset):
            # Prepare data for spatial analysis
            dataset = dataset.dropna(
                thresh=threshold_of_missing_values * len(dataset), 
                axis=1
            )
            numeric_cols = dataset.select_dtypes(
                include=['float64', 'int64']
            ).columns
            numeric_cols = numeric_cols[
                ~numeric_cols.isin(['geometry', 'latitude', 'longitude'])
            ]
            
            dataset[numeric_cols] = dataset[numeric_cols].apply(
                lambda x: x.fillna(x.mean())
            )
            
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(dataset[numeric_cols])
            standard_dataset = pd.DataFrame(
                scaled_data, 
                columns=dataset[numeric_cols].columns
            )
            
            standard_dataset['latitude'] = dataset['latitude']
            standard_dataset['longitude'] = dataset['longitude']
            
            return standard_dataset
            
        spatial_data = prepare_spatial_data(master_dataset)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            spatial_data, 
            crs="EPSG:4326", 
            geometry=gpd.points_from_xy(
                spatial_data.longitude, 
                spatial_data.latitude
            )
        )
        
        # Moran's I analysis
        weights = KNN.from_dataframe(gdf, k=8)
        moran = Moran(gdf[target_column], weights)
        local_moran = Moran_Local(gdf[target_column], weights)
        
        st.write(f"Global Moran's I: {moran.I:.3f} (p-value: {moran.p_sim:.3f})")
        
        # Map visualization
        st.subheader("Interactive Map")
        
        map_center = [
            spatial_data['latitude'].mean(), 
            spatial_data['longitude'].mean()
        ]
        
        m = folium.Map(location=map_center, zoom_start=5)
        
        # Add heatmap
        heat_data = [[row['latitude'], row['longitude'], row[target_column]] 
                    for _, row in spatial_data.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        st.components.v1.html(m._repr_html_(), height=600)

else:
    st.error("Please check your configuration settings and try again.")