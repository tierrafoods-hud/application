import streamlit as st # type: ignore
from utils.sidebar_menu import sidebar
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from utils.helper import plot_distribution_charts, replace_invalid_dates, folium_map
import os
from datetime import datetime
from utils.grids import get_grid_cell

global DATASET

@st.cache_data
def load_dataset(data):
    """
    Load a dataset from a file based on its extension. If the file has a '.gpkg' extension, load it using geopandas. Otherwise, load it using pandas.
    First checks if data exists in session state, otherwise loads from file.
    @param data - the file to load
    @return the loaded dataset
    """
    if 'processed_files' in st.session_state:
        return pd.read_csv(st.session_state['processed_files']['csv'])
    else:
        if data.name.endswith('.gpkg'):
            return gpd.read_file(data)
        else:
            return pd.read_csv(data)

@st.cache_data
def vis_missing_data(data, country_name):
    """
    Visualize missing data in a dataset using a heatmap.
    @param data - The dataset containing missing values
    @return A heatmap showing the locations of missing values in the dataset.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title(f'Missing values in the dataset for `{country_name}`')
    st.pyplot(plt)

@st.cache_data
def apply_filters(data, filters):
    """
    Apply filters to a dataset based on specified criteria.
    @param data - the dataset to filter
    @param filters - a dictionary containing filter criteria
    @return The filtered dataset based on the applied filters.
    """
    dataset = data.copy()

    # country filter
    if 'country' in filters:
        dataset = dataset[dataset['country_name'] == filters['country']]

    # bounding box filter
    if 'latitude' in dataset.columns and 'longitude' in dataset.columns and filters['bounding_box'] is not None:
        dataset = dataset[(dataset['latitude'] >= filters['bounding_box'][1]) & 
                           (dataset['latitude'] <= filters['bounding_box'][3]) & 
                           (dataset['longitude'] >= filters['bounding_box'][0]) & 
                           (dataset['longitude'] <= filters['bounding_box'][2])]
    elif 'geometry' in dataset.columns and filters['bounding_box'] is not None:
        dataset = dataset[dataset['geometry'].apply(lambda x: x.within(gpd.GeoSeries.from_wkt(filters['bounding_box']).iloc[0]))]
    
    # date range filter
    if 'date' in dataset.columns:
        dataset['date'] = dataset['date'].apply(replace_invalid_dates)
        start_date = pd.to_datetime(filters['start_date'])
        end_date = pd.to_datetime(filters['end_date'])

        dataset = dataset[(dataset['date'] >= start_date) & 
                           (dataset['date'] <= end_date)]

    return dataset

@st.cache_data
def timeseries_analysis(data, filters):
    """
    Perform time series analysis on the given dataset based on the specified filters.
    @param data - the dataset to analyze
    @param filters - a dictionary containing filters for the analysis
    @return The analyzed dataset with aggregated values based on the specified temporal method.
    """
    dataset = data.copy()

    # remove invalid dates
    if 'date' in dataset.columns:
        dataset['date'] = dataset['date'].apply(replace_invalid_dates)

        # filter the dataset
        start_date = pd.to_datetime(filters['temporal_range']['start_date'])
        end_date = pd.to_datetime(filters['temporal_range']['end_date'])
        dataset = dataset[dataset['date'] >= start_date]
        dataset = dataset[dataset['date'] <= end_date]

        timeseries_columns = filters['temporal_columns']

        if filters['temporal_method'] == "mean":
            # group by date and calculate the mean of the columns average
            dataset = dataset.groupby('date')[timeseries_columns].mean().reset_index()
        elif filters['temporal_method'] == "median":
            dataset = dataset.groupby('date')[timeseries_columns].median().reset_index()
        elif filters['temporal_method'] == "sum":
            dataset = dataset.groupby('date')[timeseries_columns].sum().reset_index()
        elif filters['temporal_method'] == "count":
            dataset = dataset.groupby('date')[timeseries_columns].count().reset_index()
    else:
        st.warning("Date column not found in the dataset. Cannot conduct timeseries analysis.")

    return dataset

@st.cache_data
def create_grids(data, cell_size=1000):
    row_ids = []
    col_ids = []
    for _, row in data.iterrows():
        lat, lon = row['latitude'], row['longitude']
        row_id, col_id = get_grid_cell(lat, lon, cell_size)
        row_ids.append(row_id)
        col_ids.append(col_id)
    data['row_id'] = row_ids
    data['col_id'] = col_ids
    return data

def show():
    global DATASET
    with st.expander("Configure your dataset", expanded=True if 'filters' not in st.session_state else False):
        st.header("Visualise your dataset")

        new_dataset = st.file_uploader("Upload your dataset", type=["gpkg", "csv"])

        if new_dataset:
            # clear processed files
            st.session_state.pop('processed_files', None)
            st.write(f"Using the dataset `{new_dataset.name}`")
            # load the dataset
            DATASET = load_dataset(new_dataset)
        
        if DATASET is None:
            st.error("No dataset found. Please upload a dataset or use the latest dataset.")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            country_column = st.text_input("Country column name", value=DEFAULT_COUNTRY_COLUMN, help="The name of the column that contains the country name in your dataset.")
        with col2:
            country_name = st.selectbox("Select a country", DATASET[country_column].unique(), help="Country is loaded from the dataset using the column name specified above.")
        
        use_bounding_box = st.checkbox("Use Bounding Box", help="Use bounding box to filter the data.")
        bounding_box = None
        if use_bounding_box:
            col1, col2 = st.columns(2)
            with col1:
                min_lon = st.number_input("Western Boundary", value=DEFAULT_BOUNDING_BOX[0], help="The western boundary of the bounding box.")
                min_lat = st.number_input("Southern Boundary", value=DEFAULT_BOUNDING_BOX[1], help="The southern boundary of the bounding box.")
            with col2:
                max_lon = st.number_input("Eastern Boundary", value=DEFAULT_BOUNDING_BOX[2], help="The eastern boundary of the bounding box.")
                max_lat = st.number_input("Northern Boundary", value=DEFAULT_BOUNDING_BOX[3], help="The northern boundary of the bounding box.")
            bounding_box = [min_lon, min_lat, max_lon, max_lat]
        
        if 'date' in DATASET.columns:
            col1, col2 = st.columns(2)
            date_col = pd.to_datetime(DATASET['date'], format='%Y-%m-%d', errors='coerce')
            
            if date_col.isnull().all():
                start_date = pd.to_datetime("1900-01-01")
                end_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            else:
                start_date = date_col.min()
                end_date = date_col.max()

            with col1:
                start_date = st.date_input("Start Date", 
                                        key='start_date',
                                        value=start_date, 
                                        help="The start date of the data to be visualised.")
            with col2:
                end_date = st.date_input("End Date", 
                                        key='end_date',
                                        value=end_date, 
                                        help="The end date of the data to be visualised.")

        submitted = st.button("Visualise", type="primary")

    if submitted:
        st.session_state['filters'] = {
            'country': country_name,
            'bounding_box': bounding_box,
            'start_date': start_date,
            'end_date': end_date,
        }
        st.session_state['submitted'] = True
    
    st.divider()

    if 'filters' in st.session_state and DATASET is not None:
        # apply filters
        DATASET = apply_filters(DATASET, st.session_state['filters'])

        st.session_state['dataset'] = DATASET

        tab1, tab2, tab3, tab4 = st.tabs(["Missing data", "Distribution charts", "Timeseries analysis", "Spatial analysis"])

        # visualise the missing data
        with tab1:
            st.subheader("Missing data")
            with st.spinner("Visualising missing data..."):
                vis_missing_data(DATASET, country_name)
        
        with tab2:
            # plot the distribution charts
            st.subheader("Distribution charts")
            st.write(f"The distribution charts are plotted based on the filtered dataset dates between `{start_date}` and `{end_date}`")
            with st.spinner("Plotting distribution charts..."):
                numeric_columns = DATASET.select_dtypes(include=['number']).columns.tolist()
                fig = plot_distribution_charts(numeric_columns, DATASET, country_name)
                st.pyplot(fig)

        with tab3:
            # timeseries analysis
            st.subheader("Timeseries analysis")
            # check if the default temporal columns are in the dataset
            valid_default_temporal_columns = [col for col in DEFAULT_TEMPORAL_COLUMNS if col in DATASET.columns]
            col1, col2 = st.columns(2)
            with col1:
                temporal_columns = st.multiselect("Columns to conduct timeseries analysis on", 
                                                DATASET.columns,
                                                default=valid_default_temporal_columns)
            with col2:
                temporal_method = st.selectbox("Method", ["mean", "median", "sum", "count"], help="The method to use for the timeseries analysis.")

            col1, col2 = st.columns(2)
            with col1:
                temporal_start_date = st.date_input("Start Date", 
                                                    key='temporal_start_date',
                                                    value=start_date, 
                                                    help="The start date of the data to be visualised.")
            with col2:
                temporal_end_date = st.date_input("End Date", 
                                                key='temporal_end_date',
                                                value=end_date, 
                                                help="The end date of the data to be visualised.")
            
            with st.spinner("Conducting timeseries analysis..."):
                # update filters
                st.session_state['filters']['temporal_columns'] = temporal_columns
                st.session_state['filters']['temporal_method'] = temporal_method
                st.session_state['filters']['temporal_range'] = {'start_date': temporal_start_date, 'end_date': temporal_end_date}

                timeseries_analysis_dataset = timeseries_analysis(DATASET, st.session_state['filters'])
                # plot line charts
                temporal_columns = st.session_state['filters']['temporal_columns']
                for column in temporal_columns:
                    start_date = st.session_state['filters']['temporal_range']['start_date']
                    end_date = st.session_state['filters']['temporal_range']['end_date']
                    method = st.session_state['filters']['temporal_method']
                    title = f"{method.capitalize()} values of `{column}` for the selected period between `{start_date}` and `{end_date}` for `{country_name}`"
                    st.write(title)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(data=timeseries_analysis_dataset, x='date', y=column, ax=ax)
                    ax.set_title(title)
                    ax.tick_params(axis='x', rotation=45)
                    st.pyplot(fig)

        with tab4:
            # data distribution on the map
            st.subheader("Point distribution on the map")
            with st.spinner("Conducting spatial analysis..."):
                st.write("Visualising the data distribution on the map")
                numeric_columns = DATASET.select_dtypes(include=['number']).columns.tolist()
                spatial_target_column = st.selectbox("Target column", numeric_columns, 
                                                     help="The column to be visualised on the map.",
                                                     key='spatial_target_column',
                                                     index=numeric_columns.index(DEFAULT_TEMPORAL_COLUMNS[0]) if DEFAULT_TEMPORAL_COLUMNS[0] in numeric_columns else 0
                                                     )
                map = folium_map(DATASET, spatial_target_column)
                st.markdown(f"""
                            The data distribution for `{spatial_target_column}` is visualised on the map.
                            The heatmap intensity is based on the magnitude of the value in the selected column 
                            at each latitude and longitude location. 
                            
                            Higher values in the selected column will result in more intense heatmap areas, 
                            and lower values will result in less intense areas.
                            """)
                st.components.v1.html(map._repr_html_(), height=450)

            st.subheader("Hotspot and Coldspot Analysis")
            with st.spinner("Conducting hotspot and coldspot analysis..."):
                country_map = None
                map_file_path = f"assets/maps/{country_name.lower()}/{country_name.lower()}.geojson"
                if os.path.exists(map_file_path):
                    country_map = gpd.read_file(map_file_path)
                else:
                    st.error(f"Country map for `{country_name}` not found. Please download the country map from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/")
                    st.stop()
                # Assign points to grids
                # dataset = create_grids(DATASET, cell_size=500)
                # spatial join country map and gdf
                gdf = gpd.GeoDataFrame(DATASET, geometry=gpd.points_from_xy(DATASET.longitude, DATASET.latitude))
                gdf = gpd.sjoin(gdf, country_map, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
                gdf[f'mean_{spatial_target_column}'] = gdf.groupby('name')[spatial_target_column].transform('mean')
                st.write(gdf)

                fig, ax = plt.subplots(figsize=(10, 6))
                gdf.plot(column=f'mean_{spatial_target_column}', cmap='viridis', legend=True, ax=ax)
                ax.axis('off')
                st.pyplot(fig)

                # # Group points by row_id and col_id, calculate mean of target column
                # grouped = dataset.groupby(['row_id', 'col_id']).agg({
                #     'longitude': list,
                #     'latitude': list,
                #     spatial_target_column: 'mean'
                # }).reset_index()

                # # Create MultiPolygon geometry for each grid cell
                # @st.cache_data
                # def create_multipoint(row):
                #     points = [Point(lon, lat) for lon, lat in zip(row['longitude'], row['latitude'])]
                #     return MultiPoint(points)

                # grouped['geometry'] = grouped.apply(create_multipoint, axis=1)
                
                # # Create GeoDataFrame with the aggregated data
                # gdf = gpd.GeoDataFrame(grouped, geometry='geometry', crs="EPSG:4326")
                # gdf = gdf[['geometry', spatial_target_column, 'row_id', 'col_id']].reset_index(drop=True).copy()
                # st.write(gdf)

if __name__ ==  "__main__":
    sidebar(title="Visualise Soil Data")
    with st.expander("About"):
        st.write("This tool allows you to visualise the soil data and its properties for a given dataset.")

    
    DEFAULT_COUNTRY_COLUMN = "country_name"
    DEFAULT_COUNTRY_NAME = "Mexico"
    DEFAULT_BOUNDING_BOX = [-117.12776, 14.5388286402, -86.811982388, 32.72083]
    WORLD_MAP = "assets/shapefiles/World_Countries_(Generalized)/World_Countries_Generalized.shp" # world map
    DEFAULT_TEMPORAL_COLUMNS = ["orgc", "tceq", "clay"]


    if not os.path.exists(WORLD_MAP):
        st.error("World map not found. Please download the world map from https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-0-countries/")
        st.stop()
    
    MAP_FILE = gpd.read_file(WORLD_MAP)

    DATASET = None
    if 'processed_files' in st.session_state:
        DATASET = st.session_state['processed_files']['csv']
        DATASET = load_dataset(DATASET)
        st.write(f"Using the latest dataset for `{st.session_state['processed_files']['country_name']}` or upload your own dataset below.")
    
    show()