import streamlit as st
from utils.sidebar_menu import sidebar
from utils.pipeline_manager import ModelPipeline, save_pipeline_to_db
from utils.helper import plot_distribution_charts
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from config.database import get_db
from dotenv import load_dotenv

load_dotenv()

# Configuration
DEFAULT_COUNTRY_NAME = "Mexico"
DEFAULT_TARGET_VARIABLES = ["orgc", "tceq"]
TEST_SIZE = 0.3
RANDOM_STATE = 42

# Database setup
db_type = os.getenv('DB_TYPE')
if not db_type:
    st.error("DB_TYPE environment variable not set")
    st.stop()

DB = get_db(db_type)

# Create required directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


def visualize_model_results(results: dict, title: str) -> None:
    """Create comprehensive model performance visualization"""
    
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    best_model = None
    best_score = float('inf')
    
    for idx, (name, result) in enumerate(results.items()):
        metrics = result['metrics']
        y_test = result['test_data']
        y_pred = result['predictions']
        
        # Track best model
        if metrics['mae'] < best_score:
            best_model = name
            best_score = metrics['mae']
        
        # Create scatter plot
        axes[idx].scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
        axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[idx].set_title(f'{name}\nMAE: {metrics["mae"]:.4f} | R²: {metrics["r2"]:.4f}', fontsize=10)
        axes[idx].set_xlabel('Actual')
        axes[idx].set_ylabel('Predicted')
        
        # Display metrics in Streamlit
        st.subheader(f"Model: {name}")
        col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="center", border=True)
        with col1:
            st.markdown(f"<h3 style='text-align: center;'>MSE: {metrics['mse']:.2f}</h3>", 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>MAE: {metrics['mae']:.2f}</h3>", 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f"<h3 style='text-align: center;'>R²: {metrics['r2']:.2f}</h3>", 
                       unsafe_allow_html=True)
    
    plt.suptitle(f'Model Performance Comparison - {title}', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show best model info
    st.success(f"Best performing model: {best_model} (MAE: {best_score:.4f})")
    
    return best_model


def create_download_button(filepath: str, filename: str) -> None:
    """Create download button for trained model"""
    try:
        with open(filepath, 'rb') as f:
            st.download_button(
                label="Download Best Model Pipeline",
                data=f,
                file_name=filename,
                mime='application/octet-stream'
            )
    except Exception as e:
        st.error(f"Error creating download button: {str(e)}")


def display_training_summary(pipeline_metadata: dict) -> None:
    """Display comprehensive training summary"""
    data = pipeline_metadata['data']
    
    st.subheader("Training Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Information:**")
        st.write(f"• Model Type: {data['model_name']}")
        st.write(f"• Target Variable: {data['target_column']}")
        st.write(f"• Total Features: {len(data['feature_columns'])}")
        st.write(f"• Numeric Features: {len(data['numeric_features'])}")
        st.write(f"• Categorical Features: {len(data['categorical_features'])}")
    
    with col2:
        st.write("**Performance Metrics:**")
        metrics = data['metrics']
        st.write(f"• Mean Squared Error: {metrics['mse']:.4f}")
        st.write(f"• Mean Absolute Error: {metrics['mae']:.4f}")
        st.write(f"• R² Score: {metrics['r2']:.4f}")
    
    with st.expander("Feature Details"):
        st.write("**Numeric Features:**")
        st.write(data['numeric_features'])
        st.write("**Categorical Features:**")
        st.write(data['categorical_features'])


@st.cache_data(ttl=3600)
def load_and_validate_data(file_obj, file_extension: str) -> pd.DataFrame:
    """Load and validate uploaded dataset"""
    try:
        if file_extension == '.csv':
            return pd.read_csv(file_obj)
        elif file_extension == '.gpkg':
            return gpd.read_file(file_obj)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()


def show():
    """Main application function"""
    
    st.header("Enhanced Model Training with Pipelines")
    
    with st.expander("About This Tool"):
        st.markdown("""
        ### Pipeline-Based Model Training
        
        This enhanced tool uses **scikit-learn pipelines** for:
        - **Integrated preprocessing**: Automatic scaling and encoding
        - **Reproducible models**: Consistent preprocessing for predictions
        - **Better organization**: All transformations stored in one object
        - **Streamlined deployment**: Single pipeline handles everything
        
        **Key Improvements:**
        - Unified pipeline approach instead of separate scalers/encoders
        - Enhanced database schema with better metadata storage
        - Improved error handling and progress tracking
        - Comprehensive model performance visualization
        """)
    
    with st.form("enhanced_model_training"):
        col1, col2 = st.columns(2)
        
        with col1:
            country_name = st.text_input(
                "Dataset Title", 
                value=DEFAULT_COUNTRY_NAME,
                help="Identifier for this training dataset"
            )
        
        with col2:
            target_column = st.selectbox(
                "Target Variable", 
                options=DEFAULT_TARGET_VARIABLES,
                index=0,
                help="Variable to predict"
            )
        
        dataset_file = st.file_uploader(
            "Upload Training Dataset", 
            type=["csv", "gpkg"],
            help="Combined dataset with soil and environmental properties"
        )
        
        advanced_settings = st.checkbox("Show Advanced Settings")
        
        if advanced_settings:
            test_size = st.slider("Test Set Size", 0.1, 0.5, TEST_SIZE, 0.05)
            random_state = st.number_input("Random State", value=RANDOM_STATE, min_value=0)
        else:
            test_size = TEST_SIZE
            random_state = RANDOM_STATE
        
        submitted = st.form_submit_button("Start Pipeline Training", type="primary")
    
    if submitted:
        if not dataset_file:
            st.error("Please upload a dataset")
            st.stop()
        
        if not target_column:
            st.error("Please select a target variable")
            st.stop()
        
        try:
            # Load dataset
            with st.spinner('Loading dataset...'):
                file_extension = os.path.splitext(dataset_file.name)[1]
                df = load_and_validate_data(dataset_file, file_extension)
                
                st.subheader("Dataset Overview")
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Initialize pipeline manager
            pipeline_manager = ModelPipeline(target_column=target_column)
            
            # Preprocessing
            st.header("Data Preprocessing")
            with st.spinner("Preprocessing dataset..."):
                processed_df = pipeline_manager.preprocess_dataset(df)
                
                if processed_df.empty:
                    st.error("Dataset is empty after preprocessing")
                    st.stop()
                
                st.success(f"Preprocessing complete! Final shape: {processed_df.shape}")
                
                # Save processed data
                output_path = f"outputs/processed_data_{target_column}.csv"
                processed_df.to_csv(output_path, index=False)
                st.info(f"Processed data saved to: {output_path}")
            
            # Distribution analysis
            st.header("Data Distribution Analysis")
            with st.spinner("Generating distribution plots..."):
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) > 1:  # Only if we have numeric columns besides target
                    fig = plot_distribution_charts(numeric_cols, processed_df, country_name)
                    st.pyplot(fig)
                else:
                    st.info("Limited numeric features for distribution analysis")
            
            # Feature preparation
            st.header("Feature Engineering")
            with st.spinner("Preparing features..."):
                # Create features dataframe (exclude target and coordinate columns)
                feature_cols = [col for col in processed_df.columns 
                               if col not in [target_column, 'longitude', 'latitude']]
                
                if not feature_cols:
                    st.error("No features available for training")
                    st.stop()
                
                X = processed_df[feature_cols]
                y = processed_df[target_column]
                
                # Create preprocessor
                preprocessor = pipeline_manager.create_preprocessor(processed_df)
                
                st.success(f"Features prepared: {len(pipeline_manager.feature_columns)} total features")
                st.write(f"• Numeric: {len(pipeline_manager.numeric_features)}")
                st.write(f"• Categorical: {len(pipeline_manager.categorical_features)}")
            
            # Model training
            st.header("Pipeline Training & Evaluation")
            
            with st.spinner("Creating and training pipelines..."):
                # Create pipelines
                pipelines = pipeline_manager.create_pipelines(X, y)
                st.info(f"Created {len(pipelines)} model pipelines")
                
                # Train and evaluate
                results = pipeline_manager.train_and_evaluate(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # Visualize results
            st.subheader("Model Performance Comparison")
            best_model_name = visualize_model_results(results, f"{country_name} - {target_column}")
            
            # Save best pipeline
            st.header("Pipeline Deployment")
            with st.spinner("Saving best pipeline..."):
                try:
                    # Save pipeline to file
                    pipeline_metadata = pipeline_manager.save_pipeline(
                        title=f"{country_name}_{target_column}"
                    )
                    
                    # Save to database
                    save_pipeline_to_db(DB, pipeline_metadata)
                    
                    st.success(f"Pipeline saved successfully!")
                    st.write(f"**File:** {pipeline_metadata['filename']}")
                    
                    # Display summary
                    display_training_summary(pipeline_metadata)
                    
                    # Download button
                    create_download_button(
                        pipeline_metadata['filepath'], 
                        pipeline_metadata['filename']
                    )
                    
                except Exception as e:
                    st.error(f"Error saving pipeline: {str(e)}")
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            st.exception(e)
    
    else:
        # Show instructions when no form is submitted
        st.info("Fill in the form above and click 'Start Pipeline Training' to begin")


if __name__ == "__main__":
    sidebar(title="Enhanced Model Training")
    show()