import streamlit as st
from utils.sidebar_menu import sidebar
from config.database import get_db
from dotenv import load_dotenv
import os, json, joblib
import pandas as pd
from utils.helper import replace_invalid_dates, folium_map
import matplotlib.pyplot as plt
import geopandas as gpd
import utils.grids as grids
import folium

load_dotenv()

@st.cache_data
def load_models():
    query = "SELECT * FROM models order by `title`"
    models = DB.fetchAllAsDict(query)
    return models

def select_model():
    models = load_models()
    model_list = {}
    for model in models:
        model_list[f"{model['title']} ({model['model_type']})"] = model
    
    # empty model_list
    if not model_list:
        st.error("No models found")
        st.stop()

    model_name = st.selectbox("Select a model", list(model_list.keys()), index=0)
    selected_model = model_list[model_name]
    model_path = selected_model["path"]
    
    if os.path.exists(model_path):

        model = joblib.load(model_path)
        return model, selected_model
    else:
        st.error(f"Model file not found at: {model_path}")
        return None

def load_scaler(scaler_path):
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

        return scaler
    else:
        st.error(f"Scaler file not found at: {scaler_path}")
        return None
    
def preprocess_data(dataset, model_features):
    # check if silt_plus_clay column is present
    if 'silt_plus_clay' not in dataset.columns:
        # Handle silt and clay columns if present
        if {'silt', 'clay'}.issubset(dataset.columns):
            dataset['silt_plus_clay'] = dataset[['silt', 'clay']].sum(axis=1, skipna=True)
            dataset.drop(columns=['silt', 'clay'], inplace=True)

    # Calculate organic matter metrics if orgc present
    if 'orgc' in dataset.columns:
        # Use vectorized operations for better performance
        organic_matter = 1.724 * dataset['orgc']
        dataset = dataset.assign(
            organic_matter=organic_matter,
            bulk_density=1.62 - 0.06 * organic_matter
        )
    else:
        dataset = dataset.assign(
            organic_matter=1.724,
            bulk_density=1.62
        )

    # check if the dataset has the same columns as the training dataset
    missing_cols = set(model_features) - set(dataset.columns)
    if missing_cols:

        st.error(f"The dataset is missing the following columns: {', '.join(missing_cols)}")
    
    # check date in the dataset
    if "date" not in dataset.columns:
        st.error("The dataset has no date column")
        st.stop()

    # replace invalid dates
    dataset['date'] = dataset['date'].apply(replace_invalid_dates)

    # select the features that are in the model
    dataset[model_features] = dataset[model_features].fillna(dataset[model_features].mean())

    return dataset

@st.cache_data
def project_grids(country, cell_data):
    
    world = gpd.read_file('assets/shapefiles/World_Countries_(Generalized)/World_Countries_Generalized.shp')

    # Ensure the shapefile is in WGS 84
    if world.crs != 'EPSG:6372':
        world = world.to_crs(epsg=6372)

    country = world[world['COUNTRY'] == country]
    
    # list of cell_id and what colour they should be
    # cell_colours = {(82, 280):'red'}

    fig, ax = grids.create_visuals(country, cell_colours=cell_data, cell_size=10000)

    return fig, ax

@st.cache_data
def init_grids(dataset):
    df_with_grids = grids.get_grid_cell_dataset(dataset)
    return df_with_grids

def show():

    st.header("Prediction tool")

    with st.expander("About"):
        st.write("""
                This tool allows users to load predictive models, upload test datasets, and visualize spatial data.
                Follow the steps below to generate and analyze predictions.
             
                Step 1: Load a predictive model
                - Select a model from the dropdown menu
                
                Step 2: Upload a test dataset
                - Click 'Upload Dataset' to upload a test dataset
                - Click 'Predict' to generate predictions

                Step 3: Visualize predictions
                - Click 'Visualize' to visualize the predictions
                - Click 'Download' to download the predictions
                """)

    model, selected_model = select_model()
    
    with st.spinner("Loading model..."):
        if model:
            scaler = load_scaler(selected_model['scaler'])
            model_features = json.loads(selected_model['features'])
            model_target = selected_model['target']
            predicted_column = f'{model_target}_predicted'

            col_names = ['date', 'latitude', 'longitude']
            col_names.extend(model_features)

            st.write(f"Selected model: {selected_model['title']}")
            if selected_model['description'] is not None:
                st.write(f"{selected_model['description']}")
            # Upload a test dataset
            st.write(f"The test dataset should have the following columns: \n{', '.join(col_names)}.")
            st.write("You can download a template dataset from the link below:")
            
            with open('assets/templates/test_dataset_template.csv', 'rb') as f:
                csv_bytes = f.read()
            
                st.download_button(
                    label="Dataset Template",
                    data=csv_bytes,
                    file_name=f"test_dataset_template.csv",
                    mime="text/csv"
                )


            uploaded_file = st.file_uploader("Upload a test dataset", type=["csv", "xlsx"])
            if uploaded_file:


                st.write(f"Uploaded file: {uploaded_file.name}")
                try:
                    if uploaded_file.type == "text/csv":
                        dataset = pd.read_csv(uploaded_file)
                    elif uploaded_file.type == "text/xlsx":
                        dataset = pd.read_excel(uploaded_file)
                    else:
                        st.error("Invalid file type. Please upload a CSV or Excel file.")
                        return
                    
                    dataset = preprocess_data(dataset, model_features)

                    st.subheader("Dataset:")
                    st.write("The pre-processsed dataset as uploaded by you is shown below")
                    st.dataframe(dataset)

                    # make predictions
                    X = dataset[model_features] #  use prediction features

                    # check if dataset has missing values
                    if X.isnull().any().any():
                        # Get columns with missing values
                        missing_cols = X.columns[X.isnull().any()].tolist()
                        st.error(f"The dataset has missing values in the following columns: {', '.join(missing_cols)}. Please check the dataset and try again.")
                        return

                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)


                    # add predictions to the dataset
                    dataset[predicted_column] = predictions

                    st.subheader("Predictions:")
                    st.write(f"The predictions for {model_target} are shown below")
                    st.dataframe(dataset)

                    # if the dataset has the target column plot the predictions vs the target
                    # if model_target in dataset.columns:
                    #     st.subheader("Predictions vs Target:")
                    #     fig, ax = plt.subplots(figsize=(10, 6))
                    #     ax.scatter(dataset[model_target], dataset[f'{model_target}_predicted'], alpha=0.7, edgecolors='k')
                    #     ax.plot([dataset[model_target].min(), dataset[model_target].max()], 
                    #         [dataset[model_target].min(), dataset[model_target].max()], 
                    #         'r--', label='Perfect Prediction')
                    #     ax.set_xlabel(f'Actual {model_target}')
                    #     ax.set_ylabel(f'Predicted {model_target}')
                    #     ax.legend()
                    #     st.pyplot(fig)
                    #     plt.close()

                    #     st.subheader("Acctuals and Predictions:")
                    #     st.dataframe(dataset[['date', model_target, f'{model_target}_predicted']])

                    # get unique upper_depth

                    
                    st.subheader("Depth Profile and Visualization")
                    soil_ranges = [
                        "0-5cm",
                        "5-15cm",
                        "15-30cm"
                    ]

                    col1, col2 = st.columns(2, gap="small", vertical_alignment="bottom")
                    with col1:
                        aggregate = st.checkbox("Aggregate", help="Aggregate the predictions by latitude and longitude ignoring the upper and lower depths", value=True)
                    with col2:
                        if not aggregate:
                            col1, col2 = st.columns(2, gap="small", vertical_alignment="bottom")
                            with col1:
                                upper_depth = st.number_input("Upper depth (in cm)", min_value=0, max_value=100, value=5)
                            with col2:
                                lower_depth = st.number_input("Lower depth (in cm)", min_value=0, max_value=100, value=15)

                    spatial_dataset = dataset.copy()

                    if aggregate:
                        # aggregate the predictions by latitude and longitude
                        predicted_mean = predicted_column + '_mean'
                        grouped = spatial_dataset.groupby(['latitude', 'longitude'])[predicted_column].mean().reset_index()
                        spatial_dataset = spatial_dataset.merge(grouped, on=['latitude', 'longitude'], how='left', suffixes=('', '_mean'))
                        spatial_dataset[predicted_column] = spatial_dataset[predicted_mean]
                        spatial_dataset.drop(columns=[predicted_mean], inplace=True)

                    else:
                        # filter the dataset by the selected upper and lower depths
                        # upper_depth = int(soil_depth.split('-')[0])
                        # lower_depth = int(soil_depth.split('-')[1][:-2])  # Remove 'cm' suffix

                        spatial_dataset = spatial_dataset[spatial_dataset['upper_depth'] == upper_depth]
                        spatial_dataset = spatial_dataset[spatial_dataset['lower_depth'] == lower_depth]


                    with st.spinner("Generating map..."):
                        # generate a folium heatmap
                        map = folium_map(spatial_dataset, predicted_column)
                        st.write("Heatmap of the predictions for the selected depth profile or aggregated predictions")
                        st.components.v1.html(map._repr_html_(), height=450)
                    
                    with st.spinner("Generating points..."):
                        # map2
                        map2 = folium.Map(location=[spatial_dataset['latitude'].mean(), spatial_dataset['longitude'].mean()], zoom_start=4)
                        datapoints = folium.map.FeatureGroup()
                        for index, row in spatial_dataset.iterrows():
                            datapoints.add_child(
                                folium.features.CircleMarker(
                                    [row['latitude'], row['longitude']],
                                    radius=5,
                                    color='red',
                                    fill=True,
                                    fill_color='blue',
                                    fill_opacity=0.6,
                                    popup=f"{predicted_column}: {row[predicted_column]:.2f}"
                                )
                            )

                        map2.add_child(datapoints)
                        st.write("Points of the dataset")
                        st.components.v1.html(map2._repr_html_(), height=450)
                    
                    # with st.spinner("Generating grids..."):
                        # generate grids dataframe
                        # df_with_grids = init_grids(dataset)                    
                        # st.dataframe(df_with_grids)

                        # Calculate mean predicted value for each grid cell
                        # grid_means = df_with_grids.groupby('cell_id')[f'{model_target}_predicted'].mean()
                        
                        # Create color mapping using a colormap
                        # norm = plt.Normalize(grid_means.min(), grid_means.max())
                        # cmap = plt.cm.YlOrRd  # Yellow-Orange-Red colormap
                        
                        # Create dictionary mapping cell_ids to colors based on predicted values
                        # cell_data = {cell_id: cmap(norm(value)) for cell_id, value in grid_means.items()}

                        # fig, ax = project_grids(DEFAULT_COUNTRY, cell_data)
                        # st.pyplot(fig)

                except Exception as e:

                    st.error(f"Error reading file: {str(e)}")
                    return

if __name__ == "__main__":


    sidebar(title="Projections")    

    db_type = os.getenv("DB_TYPE")
    DB = get_db(db_type)

    DEFAULT_COUNTRY = "Mexico"


    show()