import streamlit as st
from utils.sidebar_menu import sidebar
from config.database import get_db
from dotenv import load_dotenv
import os, json, joblib
import pandas as pd
from utils.helper import calculate_confidence_score, folium_map, calculate_horizon_fractions, calculate_SOC_stocks
import matplotlib.pyplot as plt
import geopandas as gpd
import utils.grids as grids
import folium
import numpy as np

load_dotenv()

CATEGORICAL_FEATURES = ["landcover", "zone_number"]

@st.cache_data
def load_models():
    query = "SELECT * FROM models order by last_updated desc"
    models = DB.fetchAllAsDict(query)
    return models

def select_model():
    models = load_models()
    model_list = {}
    for model in models:
        if not os.path.exists(model['path']):
            continue

        model_list[f"{model['title']} ({model['model_type']}) - {model['last_updated']}"] = model
    
    # empty model_list
    if not model_list:
        st.error("No models found")
        st.stop()

    model_name = st.selectbox("Select a model", list(model_list.keys()), index=0)
    selected_model = model_list[model_name]
    model_path = selected_model["path"]
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        if isinstance(model, dict) and 'features' in model:
            print(model['features'])
            return model, selected_model
        else:
            st.error("Invalid model format. 'features' key not found.")
            st.stop()
    else:
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
def preprocess_data(dataset, model):
    model_features = model['features']
    encoder = model['encoder']

    # check if the dataset has the same columns as the training dataset
    missing_cols = set(model_features) - set(dataset.columns)
    if missing_cols:
        st.error(f"The dataset is missing the following columns: {', '.join(missing_cols)}")

        add_missing_cols = st.button("Add missing columns", help="Click to add missing columns to the dataset with 0 values.", key="add_missing_cols")

        if add_missing_cols:
            for col in missing_cols:
                dataset[col] = 0
        else:
            st.stop()

    if set(CATEGORICAL_FEATURES).issubset(dataset.columns):
        encoded_cols = encoder.fit_transform(dataset[CATEGORICAL_FEATURES])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        df = dataset.drop(columns=CATEGORICAL_FEATURES)
        return pd.concat([df, encoded_df], axis=1)

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
            scaler = model['scaler']
            if isinstance(selected_model['features'], str):
                model_features = json.loads(selected_model['features'])
            else:
                model_features = selected_model['features']
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

            with st.spinner("Generating template..."):
                test_dataset_template = pd.DataFrame(columns=col_names)
                # save the template to a file
                test_dataset_template.to_csv('assets/templates/test_dataset_template.csv', index=False)
            
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
                    
                    with st.spinner("Pre-processing data..."):
                        dataset = preprocess_data(dataset, model)

                        st.subheader("Dataset:")
                        st.write("The pre-processsed dataset is shown below")
                        st.dataframe(dataset)

                    with st.spinner("Predicting values..."):
                        # make predictions
                        model_features = model['model_features']
                        X = dataset[model_features] #  use prediction features
                        
                        # st.write("Preview of the dataset being used for prediction.")
                        # st.dataframe(X, use_container_width=True)

                        # Check for missing values in the dataset
                        if X.isnull().any().any():
                            missing_cols = X.columns[X.isnull().any()].tolist()
                            st.error(f"The dataset has missing values in the following columns: {', '.join(missing_cols)}. Please check the dataset and try again OR fill the missing values by clicking the button below.")
                            if st.button("Fill Missing Values", help="Click to fill missing values with column means and continue with predictions."):
                                X.fillna(X.mean(), inplace=True)
                            else:
                                st.stop()

                        ML_MODEL = model['model']
                        X_scaled = scaler.transform(X)

                        # st.write("Preview of the datased being used for prediction.")
                        # st.dataframe(X_scaled, use_container_width=True)

                        predictions = ML_MODEL.predict(X_scaled)

                        # add predictions to the dataset
                        dataset[predicted_column] = predictions
                        
                        # calculate the fractions of the upper and lower depths
                        # dataset['depths_fractions'] = calculate_horizon_fractions(dataset['upper_depth'], dataset['lower_depth'])
                        # rock fragment volume
                        # dataset['rock_fragment_volume'] = np.random.uniform(1, 2, size=len(dataset)) # need to get this from the dataset

                        # calculate the SOC stocks for each row
                        # dataset['SOC_stocks_t/ha'] = dataset.apply(lambda row: calculate_SOC_stocks(
                        #     row['depths_fractions'],
                        #     row['bulk_density'],
                        #     row[predicted_column],
                        #     row['rock_fragment_volume']
                        # ), axis=1)

                        # convert the units and store in a new column
                        # dataset[f'{model_target}_t/ha'] = dataset[predicted_column] * 100

                        st.subheader("Predictions:")
                        st.write(f"The predictions for {model_target} are shown below")
                        st.dataframe(dataset)

                        confidence = calculate_confidence_score(ML_MODEL, X)
                        st.write(f"Model confidence score: {confidence:.2f}")
                        st.write("Output Score Range: Between 0 and 1 (higher = more confident)")

                    # if the dataset has the target column plot the predictions vs the target
                    if model_target in dataset.columns:
                        test_column = dataset[model_target]
                        st.subheader("Predictions vs Target:")
                        plt.figure(figsize=(10, 6))
                        plt.scatter(test_column, predictions, alpha=0.7, edgecolors='k')
                        plt.plot([test_column.min(), test_column.max()], [test_column.min(), test_column.max()], 'r--')
                        plt.xlabel(f'Actual {model_target}')
                        plt.ylabel(f'Predicted {model_target}')
                        st.pyplot(plt)
                        plt.close()

                    #     st.subheader("Acctuals and Predictions:")
                    #     st.dataframe(dataset[['date', model_target, f'{model_target}_predicted']])

                    # get unique upper_depth
                    
                    depth_profile_columns = ['latitude', 'longitude', 'upper_depth', 'lower_depth']
                    if set(depth_profile_columns).issubset(dataset.columns):
                        with st.spinner("Generating depth profile and visualization..."):
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

                                spatial_dataset = spatial_dataset[(spatial_dataset['lower_depth'] >= lower_depth) & (spatial_dataset['upper_depth'] <= upper_depth)]


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