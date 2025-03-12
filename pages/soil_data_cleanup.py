import streamlit as st
from utils.sidebar_menu import sidebar
import os
import pandas as pd
import geopandas as gpd
from utils.helper import replace_invalid_dates, preprocess_data

def show():
    with st.form("soil_data_cleanup"):
        # country name
        country_name = st.text_input("Enter the title of the dataset", value=DEFAULT_COUNTRY_NAME,
                                     help="This is the name of the country/region/title that the dataset belongs to")
        # file uploader
        dataset_file = st.file_uploader("Upload the combined data", type=["csv", "gpkg"], help="This is the combined dataset of all properties in the country")

        submitted = st.form_submit_button("Clean", type="primary")
        # set session state
        st.session_state['submitted'] = submitted
    
    if st.session_state['submitted']:
        if dataset_file is not None:
            with st.spinner('Processing uploaded file...'):
                if dataset_file.name.endswith('.csv') or dataset_file.name.endswith('.gpkg'):
                    dataset_file = dataset_file
                else:
                    st.error("Please upload a CSV or GPKG file")
                    st.stop()
        elif os.path.exists(DEFAULT_DATA_PATH):
            dataset_file = DEFAULT_DATA_PATH
        else:
            st.error("Please upload a dataset")
            st.stop()

        try:
            # if string, then it is a file path
            if isinstance(dataset_file, str):
                if dataset_file.endswith(".csv"):
                    df = pd.read_csv(dataset_file)
                elif dataset_file.endswith(".gpkg"):
                    df = gpd.read_file(dataset_file)
            elif dataset_file.name:
                if dataset_file.name.endswith(".csv"):
                    df = pd.read_csv(dataset_file)
                elif dataset_file.name.endswith(".gpkg"):
                    df = gpd.read_file(dataset_file)
            else:
                st.error("Please upload a valid dataset")
                st.stop()

            # replace invalid dates
            with st.spinner('Replacing invalid dates...'):
                df['date'] = df['date'].apply(replace_invalid_dates)

            # preprocess data
            with st.spinner('Preprocessing data...'):
                df = preprocess_data(df)

                with st.expander("Learn more about the preprocessing steps"):
                    st.write("""
                        ### Preprocessing Steps
                        
                        The preprocessing steps are as follows:
                        
                        - Replace invalid dates with random valid dates within a specified range
                        - Remove non-numeric columns
                        - Drop the complementary carbon variable to avoid data leakage
                        - Scale the data using StandardScaler
                        - Split the data into features and target variables
                        - Save the processed dataset in the session state
                        - Save the features and target variable in the session state
                        - Return the scaled dataset, features, and target variable

                    """)

                # if the dataset is empty, then stop
                if df.empty:
                    st.error(f"The dataset is empty after preprocessing {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df)
                    st.stop()

                if not df.empty:
                    st.success(f"Preprocessing complete!\n"
                    f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # save the processed dataset in the session state
            # df.to_csv(f"outputs/processed_data_{country_name}.csv", index=False)

            # download button
            st.download_button(label="Download processed dataset", 
                               data=df.to_csv(index=False), 
                               file_name=f"processed_data_{country_name}.csv")

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

if __name__ == "__main__":
    sidebar(title="Soil Data Cleanup")
    st.title("Soil Data Cleanup")
    st.write("This page is under construction. Please check back later.")

    DEFAULT_COUNTRY_NAME = "Mexico"
    DEFAULT_DATA_PATH = "outputs/clean_dataset_mexico.csv"

    show()
