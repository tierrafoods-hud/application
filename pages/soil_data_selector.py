import streamlit as st # type: ignore
import os
import pandas as pd
import geopandas as gpd
from typing import Optional, List, Tuple, Dict
import logging
import random
from utils.sidebar_menu import sidebar
from utils.helper import filter_by_country, replace_invalid_dates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_config(input_file: str, layer_prefix: str, default_layers: List[str],
    required_columns: List[str], output_path: str, country_name: str, bounding_box: Optional[List[float]]
) -> Tuple[str, str, Optional[List[float]]]:
    """
    Validate the configuration parameters for processing data.
    @param input_file: str - The input file path.
    @param layer_prefix: str - The prefix for layers.
    @param default_layers: List[str] - The default layers to consider.
    @param required_columns: List[str] - The required columns in the data.
    @param output_path: str - The output path for saving results.
    @param country_name: str - The name of the country.
    @param bounding_box: Optional[List[float]] - The bounding box coordinates.
    @return Tuple[str, str, Optional[List[float]]]: The validated country name, output path, and bounding box.
    """
    # Validate required parameters
    if not all([input_file, layer_prefix, default_layers, required_columns, output_path]):
        raise ValueError("All configuration parameters must be provided")

    # Set defaults
    country_name = country_name or DEFAULT_COUNTRY_NAME
    
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

def process_layer(input_file: str, layer: str, layer_prefix: str, country_name: str,
    required_columns: List[str], bounding_box: Optional[List[float]], depth_filters: Dict[str, float], 
    observations: pd.DataFrame, duration: Optional[Dict[str, str]]
) -> Optional[gpd.GeoDataFrame]:
    """
    Process a single layer with all filtering steps based on the provided parameters.
    @param input_file: str - The file path of the input file.
    @param layer: str - The layer to process.
    @param layer_prefix: str - The prefix for the layer.
    @param country_name: str - The name of the country for filtering.
    @param required_columns: List[str] - The list of required columns.
    @param bounding_box: Optional[List[float]] - The bounding box coordinates.
    @param depth_filters: Dict[str, float] - The depth filters.
    @param observations: pd.DataFrame - The observations data.
    @param duration: Optional[Dict[str, str]] - The duration for filtering.
    @return Optional[gpd.GeoDataFrame]: The processed GeoDataFrame or None if no data found
    """
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
    df = df.rename(columns={'value_avg': f'{layer}'})
    
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
        df['date'] = df['date'].apply(replace_invalid_dates, default_start_date=DEFAULT_START_DATE, default_end_date=DEFAULT_END_DATE)
        df = df[df['date'].notna()]
        df = df[df['date'].between(duration['start_date'], duration['end_date'])]

    if df is None or df.empty or len(df) == 0:
        logger.warning(f"No data found for {obs_str}")
        return None
    
    logger.info(f"Processed {layer.upper()}: {len(df)} records")
    return df

def show():
    st.header("Soil Data Configurations")
    
    col1, col2 = st.columns(2)
    with col1:
        input_file = st.text_input(
            "Input File Path",
            value=WOSIS_DEC_2023_PATH,
            help="Path to WoSIS GeoPackage file (local file path required due to size)"
        )

    with col2:
        layer_prefix = st.text_input(
            "Layer Prefix",
            value=PREFIX_LAYER_NAME,
            help="Prefix for layer names in the GeoPackage"
        )

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    # Geographic Filtering
    with col1:
        country_name = st.text_input("Country Name", value=DEFAULT_COUNTRY_NAME)
    with col2:
        use_bounding_box = st.checkbox("Use Bounding Box")


    bounding_box = None
    if use_bounding_box:
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Western Boundary", value=DEFAULT_BOUNDING_BOX[0])
            min_lat = st.number_input("Southern Boundary", value=DEFAULT_BOUNDING_BOX[1])
        with col2:

            max_lon = st.number_input("Eastern Boundary", value=DEFAULT_BOUNDING_BOX[2])
            max_lat = st.number_input("Northern Boundary", value=DEFAULT_BOUNDING_BOX[3])
        bounding_box = [min_lon, min_lat, max_lon, max_lat]


    # Date Range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=DEFAULT_START_DATE)
    with col2:
        end_date = st.date_input("End Date", value=DEFAULT_END_DATE)


    # Depth Filtering
    col1, col2 = st.columns(2)
    with col1:
        min_upper_depth = st.number_input("Min Upper Depth (cm)", value=0, help="This is the minimum upper depth of soil to filter the data by")
        max_upper_depth = st.number_input("Max Upper Depth (cm)", value=30, help="This is the maximum upper depth of soil to filter the data by")
    with col2:
        min_lower_depth = st.number_input("Min Lower Depth (cm)", value=0, help="This is the minimum lower depth of soil to filter the data by")
        max_lower_depth = st.number_input("Max Lower Depth (cm)", value=30, help="This is the maximum lower depth of soil to filter the data by")

    if st.button("Process Data", type="primary"):
        try:
            # Validate configuration
            country_name, output_path, bounding_box = validate_config(
                input_file, layer_prefix, DEFAULT_LAYERS, REQUIRED_COLUMNS, 
                OUTPUT_PATH, country_name, bounding_box
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
            merge_columns = [col for col in REQUIRED_COLUMNS if col != 'value_avg']
            progress_bar = st.progress(0)

            for i, layer in enumerate(DEFAULT_LAYERS):
                df = process_layer(
                    input_file, layer, layer_prefix, country_name,
                    REQUIRED_COLUMNS, bounding_box, depth_filters, observations, duration=duration
                )
                if df is not None and len(df) > 0:
                    processed_dfs.append(df)

                progress_bar.progress((i + 1) / len(DEFAULT_LAYERS))

            # Merge all processed dataframes
            if processed_dfs:
                master_df = processed_dfs[0]
                for df in processed_dfs[1:]:
                    master_df = master_df.merge(df, on=merge_columns, how='outer')
                
                # Convert to GeoDataFrame and save
                master_df = gpd.GeoDataFrame(master_df)
                output_file = os.path.join(output_path, f"{country_name}_wosis_merged.gpkg")
                master_df.to_file(output_file, driver="GPKG")
                
                # Save to CSV
                csv_file = output_file.replace(".gpkg", ".csv")
                master_df.to_csv(csv_file, index=False)

                ##  FUTURE: Save to Database
                st.session_state[f"{country_name}_soil_data"] = master_df
                st.session_state['processed_files'] = {
                    'gpkg': output_file,
                    'csv': csv_file,
                    'country_name': country_name
                }
                
                status_text.success("Processing complete! Download the data below.")
            else:
                status_text.warning("No data found for any layers")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

    # Show download buttons outside the process button logic if files are available
    if 'processed_files' in st.session_state:
        st.info(f"Download the processed data for {st.session_state['processed_files']['country_name']}.")
        files = st.session_state['processed_files']
        col1, col2 = st.columns(2)
        with col1:
            with open(files['gpkg'], 'rb') as f:
                gpkg_bytes = f.read()
            st.download_button(
                label="Download GeoPackage",
                data=gpkg_bytes,
                file_name=f"{files['country_name']}_wosis_merged.gpkg",
                mime="application/geopackage"
            )
        with col2:
            with open(files['csv'], 'rb') as f:
                csv_bytes = f.read()
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name=f"{files['country_name']}_wosis_merged.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    # Define constants
    WOSIS_DEC_2023_PATH = "./data/WoSIS_2023_December/wosis_202312.gpkg"
    PREFIX_LAYER_NAME = "wosis_202312_"
    OUTPUT_PATH = "../outputs/"
    DEFAULT_LAYERS = ["bdfiad", "bdfiod", "bdwsod", "cecph7", "cecph8", "cfvo", "clay", "ecec", "elco50", "nitkjd", "orgc", "orgm", 
                    "phaq", "phetm3", "sand", "silt", "tceq", "totc", "wv0010", "wv0033", "wv1500"]
    REQUIRED_COLUMNS = [
            'date', 'longitude', 'latitude',
            'upper_depth', 'lower_depth',
            'country_name', 'region', 'continent',
            'value_avg'
        ]
    DEFAULT_START_DATE = pd.to_datetime("1900-01-01")
    DEFAULT_END_DATE = pd.to_datetime("2023-12-31")
    DATA_TYPE_SPECIFICATIONS = {
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
    DEFAULT_COUNTRY_NAME = "Mexico"
    DEFAULT_BOUNDING_BOX = [-117.12776, 14.5388286402, -86.811982388, 32.72083]  

    # page config
    sidebar(title="Soil Data Selector")

    with st.expander("About"):

        st.write("This is a tool to help you select and process the required soil data for a specific location and time period.")
        st.write("You can select the country, the bounding box, the date range, and the depth filters to get the required data.")
        st.write("You can then use this data to train a machine learning model based on the data.")

        st.subheader("Note:")
        st.write("- The data is stored in a csv file.")
        st.write("- The data is stored in the outputs folder.") 
        st.write("- In case of invalid dates, it is replaced with a random valid date within the date range.")
    
    show()