import streamlit as st # type: ignore
import os
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def show():
    st.markdown("""
    # Data Selector
    
    Configure and process WoSIS (World Soil Information Service) data
    
    This page allows you to:
    - Set input/output paths and formats
    - Select soil property layers to extract
    - Configure geographic and depth filtering parameters
    - Specify data types for memory optimization
    """)

    # Input/Output Configuration
    st.subheader("Input/Output Configuration")
    input_file = st.text_input(
        "Input File Path",
        value="./data/WoSIS_2023_December/wosis_202312.gpkg",
        help="GeoPackage containing WoSIS data! Cannot upload file because of file size."
    )
    layer_prefix = st.text_input(
        "Layer Prefix",
        value="wosis_202312_",
        help="Prefix for layer names in the GeoPackage"
    )
    output_path = "../outputs/"

    # Required Data Layers
    st.subheader("Soil Properties")
    default_layers = "bdfiad,bdfiod,bdwsod,cecph7,cecph8,cfvo,clay,ecec,elco50,nitkjd,orgc,orgm,phaq,phetm3,sand,silt,tceq,totc,wv0010,wv0033,wv1500"
    available_layers_input = st.text_area(
        "Enter soil properties to extract (comma-separated)",
        value=default_layers,
        help="Enter soil property codes separated by commas"
    )
    
    required_layers = [layer.strip() for layer in available_layers_input.split(",")]

    # Required columns
    required_columns = [
        'date', 'longitude', 'latitude',
        'upper_depth', 'lower_depth',
        'country_name', 'region', 'continent',
        'value_avg'
    ]

    # Geographic Filtering
    st.subheader("Geographic Filtering")
    country_name = st.text_input("Country Name", value="Mexico")
    
    use_bounding_box = st.checkbox("Use Bounding Box")
    bounding_box = None
    if use_bounding_box:
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Min Longitude", value=-117.12776)
            min_lat = st.number_input("Min Latitude", value=14.5388286402)
        with col2:
            max_lon = st.number_input("Max Longitude", value=-86.811982388)
            max_lat = st.number_input("Max Latitude", value=32.72083)
        bounding_box = [min_lon, min_lat, max_lon, max_lat]

    # Date Range
    st.subheader("Time Period")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("1900-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

    # Depth Filtering
    st.subheader("Depth Filtering")
    col1, col2 = st.columns(2)
    with col1:
        min_upper_depth = st.number_input("Min Upper Depth (cm)", value=0)
        max_upper_depth = st.number_input("Max Upper Depth (cm)", value=30)
    with col2:
        min_lower_depth = st.number_input("Min Lower Depth (cm)", value=0)
        max_lower_depth = st.number_input("Max Lower Depth (cm)", value=30)

    # Data Type Specifications
    dtype_dict = {
        'date': 'str',
        'longitude': 'float32',
        'latitude': 'float32',
        'country_name': 'category',
        'region': 'category',
        'continent': 'category',
        'upper_depth': 'float32',
        'lower_depth': 'float32',
        'value_avg': 'float32'
    }

    if st.button("Process Data"):
        try:
            # Validate configuration
            country_name, output_path, bounding_box = validate_config(
                input_file, layer_prefix, required_layers, required_columns, 
                output_path, country_name, bounding_box
            )

            # Load observations once
            observations = gpd.read_file(input_file, layer=f"{layer_prefix}observations")

            # Define depth filters
            depth_filters = {
                'max_upper': max_upper_depth,
                'max_lower': max_lower_depth,
                'min_upper': min_upper_depth,
                'min_lower': min_lower_depth
            }

            # filter by date
            duration = {
                'start_date': start_date.strftime("%Y-%m-%d"),
                'end_date': end_date.strftime("%Y-%m-%d")
            }

            status_text = st.empty()
            status_text.info("Processing data...")

            # Process layers
            processed_dfs = []
            merge_columns = [col for col in required_columns if col != 'value_avg']
            progress_bar = st.progress(0)

            for i, layer in enumerate(required_layers):
                df = process_layer(
                    input_file, layer, layer_prefix, country_name,
                    required_columns, bounding_box, depth_filters, observations, duration=duration
                )
                if df is not None and len(df) > 0:
                    processed_dfs.append(df)
                progress_bar.progress((i + 1) / len(required_layers))

            # Merge all processed dataframes
            if processed_dfs:
                master_df = processed_dfs[0]
                for df in processed_dfs[1:]:
                    master_df = master_df.merge(df, on=merge_columns, how='outer')
                
                # Convert to GeoDataFrame and save
                master_df = gpd.GeoDataFrame(master_df)
                output_file = os.path.join(output_path, f"{country_name}_wosis_merged.gpkg")
                master_df.to_file(output_file, driver="GPKG")

                # save to csv
                csv_file = output_file.replace(".gpkg", ".csv")
                master_df.to_csv(csv_file, index=False)
                
                status_text.success(f"""
                Processing complete!
                - Total records: {len(master_df)}
                - Files saved:
                  - {output_file}
                  - {csv_file}
                """)
            else:
                status_text.warning("No data found for any layers")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

def validate_config(
    input_file: str,
    layer_prefix: str, 
    required_layers: List[str],
    required_columns: List[str],
    output_path: str,
    country_name: str,
    bounding_box: Optional[List[float]]
) -> Tuple[str, str, Optional[List[float]]]:
    """
    Validate configuration parameters and create output directory if needed.
    
    Returns:
        Tuple of validated country_name, output_path, and bounding_box
    """
    # Validate required parameters
    if not all([input_file, layer_prefix, required_layers, required_columns, output_path]):
        raise ValueError("All configuration parameters must be provided")

    # Set defaults
    country_name = country_name or "Mexico"
    
    # Validate bounding box if provided
    if bounding_box and len(bounding_box) != 4:
        raise ValueError("Bounding box must contain exactly 4 values: min_lat, max_lat, min_lon, max_lon")
        
    # Check input file exists
    if not os.path.exists(input_file) and not input_file.startswith(('http://', 'https://')):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    if not os.path.exists(output_path):
        output_path = os.path.join(os.path.dirname(input_file), country_name)
        os.makedirs(output_path, exist_ok=True)
        
    return country_name, output_path, bounding_box

def filter_by_country(df: pd.DataFrame, country_name: str) -> Optional[pd.DataFrame]:
    """Filter dataframe by country name"""
    return df[df['country_name'] == country_name] if country_name in df['country_name'].unique() else None

def process_layer(
    input_file: str,
    layer: str,
    layer_prefix: str,
    country_name: str,
    required_columns: List[str],
    bounding_box: Optional[List[float]],
    depth_filters: Dict[str, float],
    observations: pd.DataFrame,
    duration: Optional[Dict[str, str]]
) -> Optional[gpd.GeoDataFrame]:
    """Process a single layer with all filtering steps"""
    
    # Read layer data
    df = gpd.read_file(input_file, layer=f"{layer_prefix}{layer}")

    # Get observation description
    obs_str = f"{layer.upper()} - {observations[observations['code'] == layer.upper()]['property'].iloc[0]}" \
        if layer.upper() in observations['code'].unique() else f"Invalid {layer.upper()}"
    
    # Apply filters
    df = filter_by_country(df, country_name)
    if df is None or df.empty:
        logger.warning(f"No data found for {obs_str}")
        return None
    
    # Apply bounding box filter if specified
    if bounding_box:
        df = df[df['latitude'].between(bounding_box[1], bounding_box[3]) & df['longitude'].between(bounding_box[0], bounding_box[2])]
    
    # Select and rename columns
    df = df[required_columns].copy()
    df = df.rename(columns={'value_avg': f'{layer}_value'})
    
    # Apply depth filters
    depth_mask = (
        (df['upper_depth'] <= depth_filters['max_upper']) & 
        (df['upper_depth'] >= depth_filters['min_upper']) &
        (df['lower_depth'] <= depth_filters['max_lower']) & 
        (df['lower_depth'] >= depth_filters['min_lower'])
    )
    df = df[depth_mask]

    # filter by date
    if duration:
        df['date'] = df['date'].apply(replace_invalid_dates)
        df = df[df['date'].notna()]
        df = df[df['date'].between(duration['start_date'], duration['end_date'])]

    if df is None or df.empty or len(df) == 0:
        logger.warning(f"No data found for {obs_str}")
        return None
    
    logger.info(f"Processed {layer.upper()}: {len(df)} records")
    return df

def replace_invalid_dates(date_str):
    # Split the date string into components
    parts = date_str.split('-')
    
    # Initialize default values for year, month, and day
    year, month, day = '1900', '01', '01'
    
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

if __name__ == "__main__":
    show()