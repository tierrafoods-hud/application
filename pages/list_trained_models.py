import streamlit as st
from utils.sidebar_menu import sidebar
from config.database import get_db
from dotenv import load_dotenv
import os
import json
import pandas as pd
load_dotenv()

def get_models():
    query = "SELECT * FROM models order by last_updated desc"
    models = DB.fetchAllAsDict(query)
    return models

def show():
    # get all models from the database
    with st.spinner("Loading models..."):
        models = get_models()

        if len(models) == 0:
            st.error("No models found")
            st.stop()

        # create a table with specific columns
        for model in models:
            with st.spinner(f"Loading model {model['title']}..."):
                with st.expander(f"{model['title']}", expanded=True):
                    if isinstance(model['features'], str):
                        features = json.loads(model['features'])  # Only parse if it's a string
                    else:
                        features = model['features']  # Use it directly if it's already a list
                    st.markdown(f"### {model['title']}")
                    st.markdown(f"**Model Type:** {model['model_type']}")
                    st.markdown(f"**Features:** {', '.join(features)}")
                    st.markdown(f"**Target:** {model['target']}")
                    st.markdown(f"**Last Updated:** {model['last_updated']}")
                    
                    if model['description']:
                        st.markdown(model['description'])
                    else:
                        st.markdown("*No description provided*")
                    
                    st.markdown("#### Performance Metrics")
                    
                    if isinstance(model['metrics'], str):
                        metrics = json.loads(model['metrics'])
                    else:
                        metrics = model['metrics']
                    
                    # Create a DataFrame for the metrics table
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MSE", f"{metrics['mse']:.2f}")
                    col2.metric("MAE", f"{metrics['mae']:.2f}") 
                    col3.metric("R2", f"{metrics['r2']:.2f}")

                    model_path = model['path']
                    # scalar_path = model['scaler']

                    if not os.path.exists(model_path):
                        st.warning(f"Model file or scaler file not found for {model['title']}. Please retrain the model.")
                        continue

                    # col1, col2 = st.columns(2)
                    # download button
                    with open(model_path, 'rb') as f:
                        st.download_button(label="Download Model", data=f, file_name=f"{model['title']}.pkl", key=model['id'])
            
if __name__ == "__main__":
    # database
    db_type = os.getenv('DB_TYPE')
    DB = get_db(db_type)

    about = "This page lists all the trained models in the database. You can use these models to predict the price of a property in a given country/region/title."
    sidebar(title="List Trained Models",
            about=about,
            )
    
    st.header("List of Trained Models")
    
    show()