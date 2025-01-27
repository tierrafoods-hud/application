import streamlit as st # type: ignore
import pandas as pd
import os
from utils.display_table import display_table_data
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go # type: ignore
import joblib

# model imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# constants
threshold = 0.3
output_dir = "./outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# model list
models = {
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor()
}

def replace_invalid_dates(date_str):
    try:
        # Try direct conversion first
        return pd.to_datetime(date_str)
    except:
        # If direct conversion fails, parse components
        parts = date_str.split('-')
        
        # Validate year (required)
        if len(parts) < 1 or not parts[0].isdigit() or len(parts[0]) != 4:
            return pd.NaT
            
        # Use defaults for invalid/missing month/day
        month = parts[1].zfill(2) if len(parts) > 1 and parts[1].isdigit() and 1 <= int(parts[1]) <= 12 else '01'
        day = parts[2].zfill(2) if len(parts) > 2 and parts[2].isdigit() and 1 <= int(parts[2]) <= 31 else '01'
        
        return pd.to_datetime(f"{parts[0]}-{month}-{day}")

# side bar
with st.sidebar:
    with st.expander("Data Configuration", expanded=True):
        # file uploader
        file = st.file_uploader("Upload CSV File", type=["csv"])

        if file is not None:
            master_dataset = pd.read_csv(file)
        else:
            st.error("Please upload a CSV file to proceed.")
        
        # country name
        country_name = st.text_input("Country Name", value="Mexico")

        if file is not None:
            # target columns
            target_columns = st.multiselect("Select Target Columns", 
                                            master_dataset.columns.tolist(), 
                                            help="Select the columns to be used as target columns for the model. "
                                            "Each target column will be used to train a separate model.",
                                            max_selections=2,
                                            default=["orgc_value", "tceq_value"]
                                        )
        # test size
        test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        # random state
        random_state = st.number_input("Random State", min_value=1, max_value=100, value=42, step=1)

if file is not None:
    tab1, tab2, tab3 = st.tabs(["Model Configuration", "Model Training", "Model Evaluation"])

    with tab1:
        st.subheader(f"Data for {country_name}")
        # replace invalid dates
        master_dataset['date'] = master_dataset['date'].apply(replace_invalid_dates)
        display_table_data(master_dataset)

        # plot missing data
        st.subheader("Missing Data")
        st.bar_chart(master_dataset.isnull().sum())

        # remove missing data
        master_dataset = master_dataset.dropna(thresh=threshold * len(master_dataset), axis=1)
        # remove negative values
        master_dataset = master_dataset[master_dataset.select_dtypes(include=['number']).ge(0).all(axis=1)]

        # visualise distribution of data
        st.subheader("Distribution of Data")
        st.bar_chart(master_dataset.describe())

        numeric_columns_dataset = master_dataset.select_dtypes(include=['number']).columns

        # remove outliers using z-score
        st.subheader("Removing outliers and replacing with average of the column")
        z_scores = np.abs(stats.zscore(master_dataset[numeric_columns_dataset]))
        master_dataset = master_dataset[(z_scores < 3).all(axis=1)]
        # fill missing values with mean
        # master_dataset[numeric_columns_dataset] = master_dataset[numeric_columns_dataset].fillna(master_dataset[numeric_columns_dataset].mean())
        st.write(f"Length of dataset: {len(master_dataset)}")

        # Split and prepare datasets for each target column
        datasets = {}
        for target_column in target_columns:
            # Get dataset with non-null target values
            target_dataset = master_dataset[master_dataset[target_column].notna()]

            # scale the dataset
            scaler = MinMaxScaler()
            target_dataset[numeric_columns_dataset] = scaler.fit_transform(target_dataset[numeric_columns_dataset])
            
            # Split features and target
            X = target_dataset.drop(columns=target_columns)
            y = target_dataset[target_column]
            
            # Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Store splits in dictionary
            datasets[target_column] = {
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test
            }
            
            # Cache the splits
            st.cache_data.set(f"{target_column}_splits", datasets[target_column])
            
            # Display dataset info
            st.write(f"Dataset for {target_column}:")
            st.write(f"Training samples: {len(X_train)}")
            st.write(f"Testing samples: {len(X_test)}")

        # Cache the datasets
        st.cache_data.set(datasets, "datasets")

        st.write("Proceed to Model Training")

    with tab2:
        st.subheader("Model Training")

        st.write("Correlation Matrix")


        best_model = None
        best_model_score = float('inf')
        # Plot correlation matrices for all target columns at once
        cols = len(target_columns)
        for i, target_column in enumerate(target_columns):
            st.write(f"Correlation Matrix for {target_column}")
            # Create correlation matrix
            corr_matrix = master_dataset.corr()
            
            # Create heatmap using plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ))
            
            # Update layout
            fig.update_layout(
                title=f'Correlation Matrix for {target_column}',
                width=800,
                height=800
            )
            
            st.plotly_chart(fig)

            # get the training data and testing data
            X_train = datasets[target_column]['X_train']
            X_test = datasets[target_column]['X_test']
            y_train = datasets[target_column]['y_train']
            y_test = datasets[target_column]['y_test']

            # train the model
            for model_name, model in models.items():
                model.fit(X_train, y_train)

                # predict the model
                y_pred = model.predict(X_test)

                # evaluate the model
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                st.write(f'{model_name} - MSE: {mse}, MAE: {mae}, R2: {r2}')

                # scatter plot of predicted vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Predicted vs Actual'))
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
                fig.update_layout(title=f'Scatter Plot for {target_column} {model_name}', xaxis_title='Actual', yaxis_title='Predicted')
                st.plotly_chart(fig)

                # save the best model
                if r2 > best_model_score:
                    best_model_score = r2
                    best_model = model

        st.write(f"Best model: {best_model}")
        st.write(f"Best model score: {best_model_score}")
        #save the best model
        joblib.dump(best_model, f"{output_dir}/best_model.pkl")

    with tab3:
        st.write("Model Evaluation")
else:
    st.error("Please upload a CSV file to proceed.")


