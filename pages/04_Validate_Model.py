import streamlit as st # type: ignore
from utils.sidebar_menu import sidebar
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from utils.helper import preprocess_data, folium_map
from sklearn.preprocessing import StandardScaler
import pickle

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data(data):
    if data.name.endswith('.csv'):
        df = pd.read_csv(data)
    else:
        df = pd.read_excel(data)
    return preprocess_data(df)

if __name__ == "__main__":
    sidebar(title="Model Validation")

    st.title("Model Validation Tool")
    
    with st.expander("About this tool"):
        st.write("""
                 This tool allows you to validate trained models on new data to:
                 - Test model performance on unseen data
                 - Compare predicted vs actual values
                 - Calculate key performance metrics
                 - Visualize prediction accuracy
                 
                 To use:
                 1. Select a trained model
                 2. Upload validation data
                 3. Click 'Validate Model' to see results
                 """)
    
    # Model selection with error handling
    try:
        # Cache model loading
        @st.cache_resource
        def load_model(model_path):
            try:
                return joblib.load(model_path)
            except (ValueError, TypeError) as e:
                # Handle numpy random state loading error
                with open(model_path, 'rb') as f:
                    return pickle.load(f)

        list_of_models = [f for f in os.listdir("models") if f.endswith('.pkl')]
        if not list_of_models:
            st.error("No trained models found in models directory")
        else:
            model_name = st.selectbox("Select a trained model", list_of_models)
            model = load_model(f"models/{model_name}")

            # Extract target variable from model name
            model_target_var = model_name.split('_')[1].split('.')[0].lower()

            # File upload with progress bar
            data = st.file_uploader("Upload validation data", type=["csv", "xlsx", "xls"])
            if data is not None:
                with st.spinner('Loading and preprocessing data...'):

                    df = load_and_preprocess_data(data)

                    # Get required features from model
                    required_features = st.session_state['_features']

                    # print(required_features)
                    # print(model.feature_names_in_)

                    # Validate features
                    missing_features = [feat for feat in required_features if feat not in df.columns]
                    if missing_features:
                        st.error(f"Missing required features: {', '.join(missing_features)} the columns will be created with `zero` values thus the model may not provide best results. To ensure the model is efficient, the columns should be present in the dataset.")
                        # add the columns to the dataframe
                        for feat in missing_features:
                            df[feat] = 0

                    # Show data preview in expandable section
                    with st.expander("Preview validation data"):
                        st.dataframe(df)
                        st.text(f"Dataset shape: {df.shape}")

                    st.info(f"Target variable: {model_target_var}")

                    if st.button("Validate Model", type="primary"):
                        with st.spinner("Validating model..."):
                            # Prepare data - ensure all required features are present in correct order
                            X = pd.DataFrame()
                            for feature in required_features:
                                X[feature] = df[feature].values
                            
                            # Scale features using same scaler as training
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X)
                            
                            # Generate predictions
                            y_pred = model.predict(X_scaled)

                            # Add predictions to original dataset
                            pred_col_name = f"{model_target_var}_predicted"
                            if pred_col_name in df.columns:
                                df = df.drop(columns=[pred_col_name])
                            df[pred_col_name] = y_pred

                            # Only calculate and display metrics if target variable exists
                            if pred_col_name in df.columns:
                                y_true = df[pred_col_name].values
                                
                                # Calculate metrics
                                mse = mean_squared_error(y_true, y_pred)
                                mae = mean_absolute_error(y_true, y_pred)
                                r2 = r2_score(y_true, y_pred)
                                rmse = np.sqrt(mse)
                                
                                # Display metrics
                                col1, col2, col3, col4 = st.columns(4)
                                col1.metric("RMSE", f"{rmse:.4f}")
                                col2.metric("MAE", f"{mae:.4f}")
                                col3.metric("MSE", f"{mse:.4f}")
                                col4.metric("R² Score", f"{r2:.4f}")
                                
                                # # Create scatter plot
                                # fig, ax = plt.subplots(figsize=(10, 6))
                                # ax.scatter(y_true, y_pred, alpha=0.5)
                                
                                # # Plot perfect prediction line
                                # min_val = min(min(y_true), min(y_pred))
                                # max_val = max(max(y_true), max(y_pred))
                                # ax.plot([min_val, max_val], [min_val, max_val], 
                                #        'r--', label='Perfect Prediction')
                                
                                # ax.set_title(f'Model Predictions vs Actual Values\n{model_target_var.upper()}')
                                # ax.set_xlabel('Actual Values')
                                # ax.set_ylabel('Predicted Values')
                                # ax.legend()
                                
                                # st.pyplot(fig)
                                
                                # # Prepare results DataFrame
                                # results_df = pd.DataFrame({
                                #     'Actual': y_true,
                                #     'Predicted': y_pred,
                                #     'Absolute_Error': np.abs(y_true - y_pred),
                                #     'Percentage_Error': np.abs((y_true - y_pred) / y_true) * 100
                                # })
                                
                                # Original results download button
                                # st.download_button(
                                #     label="Download Validation Results Summary",
                                #     data=results_df.to_csv(index=False),
                                #     file_name=f"validation_results_{model_target_var}.csv",
                                #     mime="text/csv"
                                # )

                            # Add download button for complete dataset
                            st.download_button(
                                label="Download Complete Dataset with Predictions",
                                data=df.to_csv(index=False),
                                file_name=f"dataset_with_predictions_{model_target_var}.csv",
                                mime="text/csv"
                            )

                            st.subheader("Preview of the predicted data")
                            st.dataframe(df)

                            st.subheader("Map of the predicted data")
                            map = folium_map(df, pred_col_name, zoom_start=6)
                            st.components.v1.html(map._repr_html_(), height=450)
    except Exception as e:

        st.error(f"An error occurred: {str(e)}")
        st.info("Please ensure you have trained models available and valid input data")
