import streamlit as st # type: ignore
from utils.sidebar_menu import sidebar
import pandas as pd
import geopandas as gpd
from utils.helper import plot_distribution_charts, replace_invalid_dates, preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# migration file for models table
from config.database import get_db
from dotenv import load_dotenv
import json

load_dotenv()

@st.cache_data(ttl=600)
def split_and_scale_data(df, target_variable):
    """
    Split the data into features (X) and target variables (y), scale the features using StandardScaler, and return the scaled features and target variables.
    @param df - The dataframe containing the data
    @param target_variable - The target variable to predict
    @return X_scaled - The scaled features
    @return y - The target variables
    """
    # save the dataset to csv
    df.to_csv(f"outputs/processed_data_{target_variable}.csv", index=False)

    # save in session state
    st.session_state[f'{target_variable}_dataset'] = df.copy()

    # drop non-numeric columns
    df = df.select_dtypes(include=['number']).copy()

    # split data into features and target
    drop_columns = [target_variable, 'latitude', 'longitude']

    # Drop the complementary carbon variable to avoid data leakage
    # This is required to train models without the other value since they are highly correlated
    if target_variable == 'orgc':
        drop_columns.append('tceq')
    elif target_variable == 'tceq':
        drop_columns.append('orgc')

    # save the features and target variable
    st.session_state['_features'] = df.drop(columns=drop_columns, errors='ignore')
    st.session_state['_target'] = df[target_variable]

    X = df.drop(columns=drop_columns, errors='ignore') # ignore error if target_variables is not in the dataset
    y = df[target_variable]

    # scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # save the scalar
    st.session_state['_scaler'] = scaler

    # create scaled dataset
    scaled_dataset = pd.DataFrame(X_scaled, columns=df.select_dtypes(include=['number']).columns.drop(drop_columns))
    scaled_dataset[target_variable] = y.values

    # save in session state
    st.session_state[f'{target_variable}_scaled_dataset'] = scaled_dataset

    return scaled_dataset, X_scaled, y

@st.cache_data(ttl=600)
def save_model_to_db(title, model_type, features, target, scaler, metrics, model_path):
    """
    Save model metadata to database
    """
    model_data = {
        'title': title,
        'model_type': model_type,
        'features': features.columns.tolist(),
        'target': target.name,
        'scaler': scaler,
        'metrics': metrics,
        'path': model_path
    }
    
    query = """
    INSERT INTO models (title, model_type, features, target, scaler, metrics, path) 
    VALUES (%(title)s, %(model_type)s, %(features)s, %(target)s, %(scaler)s, %(metrics)s, %(path)s)
    """
    model_data['features'] = json.dumps(model_data['features'])
    model_data['metrics'] = json.dumps(model_data['metrics'])
    DB.create(query, model_data)
    return model_data


def train_model(X_scaled, y, model):
    """
    Train and evaluate a model on scaled data.
    @param X_scaled - The scaled features
    @param y - The target variables 
    @param model - The model to train
    @return Tuple of (mse, mae, r2, y_test, y_pred) evaluation metrics and test data
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Train model and get predictions
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate and return all evaluation metrics at once
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred)
    }
    
    return metrics, y_test, y_pred

def validate_model(models, features, target, title, save_model=False):
    """
    Validate multiple models and visualize their performance.
    @param models - Dict of model name to model instance
    @param features - The features to use for validation
    @param target - The target variable to predict
    @param title - The title for plots and saved model
    @param save_model - Whether to save the best performing model
    @return The best performing model based on MAE
    """
    with st.spinner('Validating models...'):
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        best_model = None
        best_model_type = None
        best_score = float('inf')

        # Create subplot grid
        fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
        
        # Train and evaluate each model
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Training and evaluating {name}...')
            
            metrics, y_test, y_pred = train_model(features, target, model)
            
            # Track best model
            if metrics['mae'] < best_score:
                best_model = model
                best_model_type = name
                best_score = metrics['mae']

            # Log metrics
            st.markdown(f"### {name} metrics:\n"
                    f"- MSE: {metrics['mse']:.4f}\n"
                    f"- MAE: {metrics['mae']:.4f}\n"
                    f"- R2: {metrics['r2']:.4f}")
            
            # plot individual model
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            plt.title(f'{name}\n(MSE: {metrics["mse"]:.4f}, MAE: {metrics["mae"]:.4f}, R2: {metrics["r2"]:.4f})', fontsize=10)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            st.pyplot(plt)
            plt.close()

            # Plot predictions vs actuals
            ax = axes[i]
            ax.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
            ax.set_title(f'{name}\n(MSE: {metrics["mse"]:.4f}, MAE: {metrics["mae"]:.4f}, R2: {metrics["r2"]:.4f})', fontsize=10)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')

            # Update progress
            progress_bar.progress((i + 1) / len(models))

        # Save best model if requested
        if save_model and best_model:
            status_text.text('Saving best model...')
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{title}_best_model.pkl'
            st.session_state[title] = best_model
            joblib.dump(best_model, model_path)

            # save the scalar to storage
            scalar_path = f'models/{title}_scaler.pkl'
            joblib.dump(st.session_state['_scaler'], scalar_path)
            
            # Save model metadata to database
            model_data = save_model_to_db(
                title,
                best_model_type,
                st.session_state['_features'],
                st.session_state['_target'],
                scalar_path,
                metrics,
                model_path


            )
            
            st.success(f"Best model saved to: {model_path} and database")
            
            # Add download button
            with open(model_path, 'rb') as f:
                st.download_button(
                    label="Download the best model",
                    data=f,
                    file_name=f'{title}_best_model.pkl',
                    mime='application/octet-stream'
                )

        # Finalize and show plot
        plt.suptitle(f'Model Performance Comparison for {title}')
        plt.tight_layout()
        plt.figtext(0.5, 0.005,
                    'Scatter plots comparing predicted vs actual values. Red dashed line shows perfect predictions.',
                    ha='center', fontsize=10)
        st.pyplot(fig)
        plt.close()

        # Clear temporary status elements
        status_text.empty()
        progress_bar.empty()

        st.success("Model validation complete!")

        return best_model

def show():
    # country name
    country_name = st.text_input("Enter the title of the dataset", value=DEFAULT_COUNTRY_NAME, help="This is the name of the country/region/title that the dataset belongs to")
    # file uploader
    dataset_file = st.file_uploader("Upload the combined data", type=["csv", "gpkg"], help="This is the combined dataset of all properties in the country")
    
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
        elif 'name' in dataset_file:
            if dataset_file.name.endswith(".csv"):
                df = pd.read_csv(dataset_file)
            elif dataset_file.name.endswith(".gpkg"):
                df = gpd.read_file(dataset_file)
        else:
            st.error("Please upload a valid dataset")
            st.stop()

        # replace invalid dates
        df['date'] = df['date'].apply(replace_invalid_dates)

        ######### SECTION 1: PREPROCESSING ANALYSIS #########
        st.header("Preprocessing Analysis")
        with st.spinner("Processing dataset..."):
            st.subheader(f"Dataset for {country_name}")
            st.dataframe(df)

            with st.spinner("Plotting missing values..."):
                # plot missing values
                st.write("""
                    An illustration of missing values in the dataset. This will help you understand the viability of the dataset for model training and also give you and idea
                            if the dataset is clean or not. For an optimal model, the dataset should have minimal missing values.
                """)
                plt.figure(figsize=(10, 6))
                missing_values = df.isnull().sum()
                plt.bar(range(len(missing_values)), missing_values)
                plt.xticks(range(len(missing_values)), missing_values.index, rotation=45, ha='right')
                plt.title(f'Missing values in the {country_name} dataset')
                plt.ylabel('Number of missing values')
                plt.tight_layout()
                st.pyplot(plt)

            with st.spinner("Removing missing values..."):
                df = preprocess_data(df)
                if df is not None:
                    st.success(f"Preprocessing complete!\n"
                    f"Final dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
                
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
        
        ######### SECTION 2: DISTRIBUTION ANALYSIS #########
        st.header("Distribution Analysis")
        with st.spinner("Plotting distribution charts..."):
            st.write("""
                An illustration of the distribution of numeric columns in the dataset. Viewing the distribution helps:

                - Identify if variables follow normal/skewed distributions
                - Detect potential outliers or unusual patterns
                - Understand the range and spread of values
                - Determine if data transformations may be needed for modeling
                - Ensure the data is suitable for the chosen regression models
            """)

            with st.spinner("Plotting distribution charts..."):
                fig = plot_distribution_charts(df.select_dtypes(include=['number']).columns, df, "Distribution of numeric columns")
                st.pyplot(fig)

        ######### SECTION 3: MODEL CONFIGURATION #########
        st.header("Build Regression Model")
        with st.expander("Instructions"):
            st.write("""
                ### Model Training Instructions
                
                Select a target variable to predict:
                
                - Multiple regression models will be trained and evaluated (Random Forest, Gradient Boosting, etc.)
                - The remaining variables in the dataset will be used as predictors/features
                - The best performing model will be automatically selected based on validation metrics
                - You'll see detailed model performance metrics and feature importance analysis
                - The optimal model will be saved for future predictions
                
                __Recommended target variable:__ soil properties like organic carbon (orgc) or total carbon equivalent (tceq)
                
                __Note:__ Ensure your dataset has sufficient samples and minimal missing values for reliable model training.
                """)
        
        target_variable = st.selectbox("Select the target variable", 
                                        options=df.columns, 
                                        index=df.columns.get_loc("orgc") if "orgc" in df.columns else 0,
                                        help="Select the target variable to predict")
        if target_variable is None:
            st.error("Please select a target variable")
            st.stop()

        if st.button("Begin Model Training", type="primary"):
            # scale and split the dataset
            scaled_dataset, X_scaled, y = split_and_scale_data(df, target_variable)

            # preview the scaled dataset
            st.write("""
                A preview of the scaled dataset.
            """)
            st.dataframe(scaled_dataset)

            # validate models
            model_title = f"{country_name}_{target_variable}"
            validate_model(MODELS_LIST, X_scaled, y, model_title, save_model=True)

    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")

if __name__ ==  "__main__":
    sidebar(title="Build Regression Model")

    with st.expander("About"):
        st.write("""
                 This tool allows you to build a regression model using the combined dataset of soil data and weather data.

                 - Ensure the data is generated using the `Soil Data Selector` and parsed using the `Weather Data Selector` tool for best results.
                 """)

    MODELS_LIST = {
        'Random Forest Regressor': RandomForestRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(),
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors Regressor': KNeighborsRegressor()
    }
    DEFAULT_DATA_PATH = "data/mexico_combined_data.csv"
    DEFAULT_COUNTRY_NAME = "Mexico"
    DEFAULT_TARGET_VARIABLES = ["orgc"]
    TEST_SIZE = 0.3
    RANDOM_STATE = 42

    # create required folders
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # database
    db_type = os.getenv('DB_TYPE')
    DB = get_db(db_type)

    show()