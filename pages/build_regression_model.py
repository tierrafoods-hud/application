import streamlit as st # type: ignore
from utils.sidebar_menu import sidebar
import pandas as pd
import geopandas as gpd
from utils.helper import plot_distribution_charts, replace_invalid_dates, preprocess_data
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# migration file for models table
from config.database import get_db
from dotenv import load_dotenv
import json
from sklearn.preprocessing import OneHotEncoder
import time

load_dotenv()

MODELS_LIST = {
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(),
    'Linear Regression': LinearRegression(),
    'Support Vector Regressor': SVR(),
    'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
    'MLP Regressor (neural network)': MLPRegressor()
}

DEFAULT_DATA_PATH = "" #"data/mexico_combined_data.csv"
DEFAULT_COUNTRY_NAME = "Mexico"
DEFAULT_TARGET_VARIABLES = ["orgc"]
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Define numeric and categorical feature columns
# NUMERIC_FEATURES = None #['upper_depth','lower_depth','silt','clay','elcosp','phaq','sand','temperature','precipitation']
CATEGORICAL_FEATURES = ['landcover', 'zone_number']
TARGET_COLUMN = None
DROP_COLUMNS = ['country_name', 'region', 'continent', 'ecoregion_type', 'zone_name']

# create required folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# database
db_type = os.getenv('DB_TYPE')
DB = get_db(db_type)

def preprocessing_analysis(df, target_variable, drop_columns=[]):
    def drop_highly_correlated(df, target_variable):
        if target_variable == 'orgc':
            return df.drop(columns=['tceq'], errors="ignore")
        elif target_variable == 'tceq':
            return df.drop(columns=['orgc'], errors="ignore")
        return df
    
    dataset = df.copy()
    initial_rows = len(dataset)
    st.write(f"Before dropping rows: {dataset.shape}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    dataset = drop_highly_correlated(dataset, target_variable)

    # Drop columns with more than 70% missing values
    status_text.text("Dropping columns with more than 70% missing values...")
    threshold = 0.7 * len(dataset)
    cols_to_drop = [col for col in dataset.columns if dataset[col].isna().sum() > threshold and col != target_variable]
    dataset.drop(columns=cols_to_drop + drop_columns, errors='ignore', inplace=True)
    progress_bar.progress(0.2)

    # Check if target variable is in the dataset
    status_text.text("Checking if target variable is in the dataset...")
    if target_variable not in dataset.columns:
        st.error(f"The target variable {target_variable} is not in the dataset")
        progress_bar.empty()
        status_text.empty()
        st.stop()

    # Drop rows with missing target variable
    status_text.text("Dropping rows with missing target variable...")
    dataset.dropna(subset=[target_variable], inplace=True)
    st.warning(f"Removed {initial_rows - len(dataset)} rows due to missing values in target column. New shape: {dataset.shape}")
    initial_rows = len(dataset)
    progress_bar.progress(0.4)

    # Drop duplicate rows
    status_text.text("Dropping duplicate rows...")
    dataset.drop_duplicates(inplace=True)
    st.warning(f"Removed {initial_rows - len(dataset)} rows as duplicates. New shape: {dataset.shape}")
    initial_rows = len(dataset)
    progress_bar.progress(0.5)

    # Remove rows with missing values
    status_text.text("Dropping rows with missing values...")
    dataset.dropna(inplace=True)
    st.warning(f"Removed {initial_rows - len(dataset)} rows with missing values. New shape: {dataset.shape}")
    progress_bar.progress(0.6)

    # Convert categorical columns to string
    dataset[CATEGORICAL_FEATURES] = dataset[CATEGORICAL_FEATURES].astype(str)
    progress_bar.progress(0.8)

    # Replace invalid zone_number values if roman numerals with integers
    if 'zone_number' in dataset.columns:
        status_text.text("Replacing invalid zone_number values...")
        dataset = dataset[~dataset['zone_number'].isin(['VII', 'VI'])]
    progress_bar.progress(1.0)

    st.write(f"Final dataset shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")
    status_text.success("Preprocessing complete!")

    # remove after 3 secs
    time.sleep(2)
    status_text.empty()
    progress_bar.empty()

    dataset.reset_index(drop=True, inplace=True)
    
    return dataset

def split_and_scale_data(df, target_variable):
    def save_to_session(key, value):
        st.session_state[key] = value

    def preprocess_categorical(df, categorical_features):
        # df[categorical_features] = df[categorical_features].astype(str)
        encoder = OneHotEncoder(drop='first', sparse_output=False)
        encoded_cols = encoder.fit_transform(df[categorical_features])
        encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_features))
        df = df.drop(columns=categorical_features)
        return pd.concat([df, encoded_df], axis=1), encoder

    df.to_csv(f"outputs/processed_data_{target_variable}.csv", index=False)
    save_to_session(f'{target_variable}_dataset', df.copy())

    if set(CATEGORICAL_FEATURES).issubset(df.columns):
        df, encoder = preprocess_categorical(df, CATEGORICAL_FEATURES)
        save_to_session('_encoder', encoder)

    NUMERIC_FEATURES = [col for col in df.select_dtypes(include=['number']).columns.tolist() if col != target_variable]
    df = df[NUMERIC_FEATURES + [target_variable]]

    X = df.drop(columns=[target_variable], errors='ignore')
    y = df[target_variable]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    save_to_session('_scaler', scaler)

    scaled_dataset = pd.DataFrame(X_scaled, columns=X.columns)
    scaled_dataset[target_variable] = y.values

    save_to_session(f'{target_variable}_scaled_dataset', scaled_dataset)
    save_to_session('_features', X)
    save_to_session('_target', y)

    return scaled_dataset, X_scaled, y

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
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    best_model = None
    best_model_type = None
    best_score = float('inf')

    # Create subplot grid
    fig, axes = plt.subplots(1, len(models), figsize=(20, 5))
    
    try:
        # Train and evaluate each model
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Training and evaluating {name}...')
            
            metrics, y_test, y_pred = train_model(features, target, model)
            
            # Track best model
            if metrics['mae'] < best_score:
                best_model = model
                best_model_type = name
                best_score = metrics['mae']

            with st.spinner(f"Plotting {name} model..."):
                st.subheader(f"Model: {name}")
                # plot metrics
                col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="center", border=True)
                with col1:
                    st.markdown(f"<h3 style='text-align: center;'>MSE: {metrics['mse']:.2f}</h3>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<h3 style='text-align: center;'>MAE: {metrics['mae']:.2f}</h3>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<h3 style='text-align: center;'>R2: {metrics['r2']:.2f}</h3>", unsafe_allow_html=True)

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

        with st.spinner(f"Plotting {name} predictions..."):
            # Finalize and show plot
            plt.suptitle(f'Model Performance Comparison for {title}')
            plt.tight_layout()
            plt.figtext(0.5, 0.005,
                        'Scatter plots comparing predicted vs actual values. Red dashed line shows perfect predictions.',
                        ha='center', fontsize=10)
            st.pyplot(fig)
            plt.close()

        # Save best model if requested
        if save_model and best_model:
            status_text.text('Saving best model...')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_name = f"{title}_best_model_{timestamp}.pkl"
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{model_name}'
            st.session_state[title] = best_model
            joblib.dump(best_model, model_path)

            # save the scalar to storage
            scalar_path = f'models/{title}_scaler_{timestamp}.pkl'
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
            
            status_text.success(f"Best model saved to: {model_path} and database")
            
            # Add download button
            with open(model_path, 'rb') as f:
                st.download_button(
                    label="Download the best model",
                    data=f,
                    file_name=f'{model_name}',
                    mime='application/octet-stream'
                )

            st.write("About the model:")
            st.write(f"Model type: {best_model_type}")
            st.write(f"Model features: {st.session_state['_features'].columns.tolist()}")
            st.write(f"Model target: {st.session_state['_target'].name}")
            st.write(f"Model metrics: {metrics}")

        status_text.success("Model validation complete!")

        time.sleep(2)
        progress_bar.empty()
        status_text.empty()

        return best_model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        st.stop()

def show():
    
    with st.form("train_model"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # country name
            country_name = st.text_input("Enter the title of the dataset", value=DEFAULT_COUNTRY_NAME,
                                     help="This is the name of the country/region/title that the dataset belongs to")
        with col2:
            # target column
            TARGET_COLUMN = st.selectbox("Select the target variable", 
                                            options=['orgc', 'tceq'], 
                                            index=0,
                                            help="Select the target variable to predict")
        # file uploader
        dataset_file = st.file_uploader("Upload the combined data", type=["csv", "gpkg"], help="This is the combined dataset of all properties in the country including temperature, precipitation, ecoregions and soil properties")
        
        submitted = st.form_submit_button("Begin Model Training", type="primary")
        
        # set session state
        st.session_state['submitted'] = submitted
    
    if st.session_state['submitted']:
        if TARGET_COLUMN is None:
            st.error("Please select a target variable")
            st.stop()

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

            with st.spinner('Preprocessing data...'):
                
                with st.expander("Learn more about the preprocessing steps"):
                    st.write("""
                        ### Preprocessing Steps
                        
                        The preprocessing steps are as follows:
                        
                        - Drop rows with invalid dates
                        - Drop columns with more than 70% missing values
                        - Check if the target variable is in the dataset
                        - Drop rows with missing target variable
                        - Drop duplicate rows
                        - Final dataset shape is displayed

                    """)
                
                df = preprocessing_analysis(df, TARGET_COLUMN, DROP_COLUMNS)

                st.subheader("Preprocessed Dataset")
                st.dataframe(df, use_container_width=True)

                # if the dataset is empty, then stop
                if df.empty:
                    st.error(f"The dataset is empty after preprocessing {df.shape[0]} rows × {df.shape[1]} columns")
                    st.dataframe(df)
                    st.stop()
            
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
                    fig = plot_distribution_charts(df.select_dtypes(include=['number']).columns.tolist(), df, country_name)
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

            # scale and split the dataset
            with st.spinner("Scaling and splitting the dataset..."):
                scaled_dataset, X_scaled, y = split_and_scale_data(df, TARGET_COLUMN)

            # preview the scaled dataset
            st.write("""
                A preview of the scaled dataset.
            """)
            st.dataframe(scaled_dataset, use_container_width=True)

            # validate models
            model_title = f"{country_name}_{TARGET_COLUMN}"
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

    show()