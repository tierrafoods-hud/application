import streamlit as st
from utils.sidebar_menu import sidebar
from utils.pipeline_manager import load_pipelines_from_db, calculate_pipeline_confidence, validate_pipeline_input
from utils.helper import folium_map
from config.database import get_db
from dotenv import load_dotenv
import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

load_dotenv()

# Database setup
db_type = os.getenv('DB_TYPE')
if not db_type:
    st.error("DB_TYPE environment variable not set")
    st.stop()

DB = get_db(db_type)


@st.cache_data(ttl=600)
def load_available_pipelines():
    """Load and cache available pipeline models"""
    try:
        pipelines = load_pipelines_from_db(DB)
        return [p for p in pipelines if os.path.exists(p['pipeline_path'])]
    except Exception as e:
        st.error(f"Error loading pipelines: {str(e)}")
        return []


def create_pipeline_selector():
    """Create model selection interface"""
    pipelines = load_available_pipelines()
    
    if not pipelines:
        st.error("No trained pipelines found. Please train a model first.")
        st.stop()
    
    # Create display options
    display_options = {}
    for pipeline in pipelines:
        display_name = f"{pipeline['title']} ({pipeline['model_type']}) - {pipeline['target']}"
        if pipeline.get('updated_at'):
            display_name += f" - {pipeline['updated_at']}"
        display_options[display_name] = pipeline
    
    selected_display = st.selectbox(
        "Select Trained Pipeline",
        options=list(display_options.keys()),
        help="Choose a previously trained model pipeline for predictions"
    )
    
    return display_options[selected_display]


def load_pipeline_model(pipeline_path: str):
    """Load pipeline model from file"""
    try:
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")
        
        pipeline_data = joblib.load(pipeline_path)
        
        # Handle both old and new format
        if 'pipeline' in pipeline_data:
            return pipeline_data
        else:
            # Old format compatibility
            st.warning("Loading legacy model format. Consider retraining for better performance.")
            return {
                'pipeline': pipeline_data.get('model'),
                'feature_columns': pipeline_data.get('features', []),
                'target_column': pipeline_data.get('target', {}).get('name', 'unknown'),
                'model_name': 'Legacy Model'
            }
    
    except Exception as e:
        st.error(f"Error loading pipeline: {str(e)}")
        st.stop()


def generate_template_dataset(pipeline_metadata: dict) -> pd.DataFrame:
    """Generate template dataset for user download"""
    feature_columns = json.loads(pipeline_metadata['feature_columns']) if isinstance(
        pipeline_metadata['feature_columns'], str) else pipeline_metadata['feature_columns']
    
    # Add coordinate columns for spatial data
    template_columns = ['latitude', 'longitude'] + feature_columns
    
    return pd.DataFrame(columns=template_columns)


def validate_prediction_data(df: pd.DataFrame, pipeline_data: dict) -> pd.DataFrame:
    """Validate and prepare data for prediction"""
    
    # Check pipeline compatibility
    is_valid, missing_features = validate_pipeline_input(df, pipeline_data)
    
    if not is_valid:
        st.error(f"Dataset missing required features: {', '.join(missing_features)}")
        
        if st.button("Add Missing Features (filled with zeros)", 
                     help="Add missing columns with zero values to proceed"):
            for feature in missing_features:
                df[feature] = 0
            st.success("Missing features added. Please review and update values if needed.")
            st.rerun()
        else:
            st.stop()
    
    # Handle data types properly
    categorical_features = ['landcover', 'zone_number']
    
    for col in df.columns:
        if col in categorical_features:
            # Ensure categorical columns are strings
            df[col] = df[col].astype(str)
        elif col not in ['latitude', 'longitude']:
            # For numeric columns, convert to numeric and handle errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values after type conversion
    if df.isnull().any().any():
        null_cols = df.columns[df.isnull().any()].tolist()
        st.warning(f"Missing values detected in: {', '.join(null_cols)}")
        
        fill_method = st.radio(
            "How to handle missing values?",
            ["Fill with column means", "Fill with zeros", "Stop and fix manually"],
            index=0
        )
        
        if fill_method == "Fill with column means":
            # Fill numeric columns with means, categorical with mode
            for col in df.columns:
                if col in categorical_features:
                    # Fill categorical with most common value
                    if df[col].isnull().any():
                        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown'
                        df[col] = df[col].fillna(mode_val)
                else:
                    # Fill numeric with mean
                    if df[col].dtype in ['int64', 'float64'] and df[col].isnull().any():
                        df[col] = df[col].fillna(df[col].mean())
        elif fill_method == "Fill with zeros":
            # Fill with appropriate defaults
            for col in df.columns:
                if col in categorical_features:
                    df[col] = df[col].fillna('unknown')
                else:
                    df[col] = df[col].fillna(0)
        else:
            st.stop()
    
    return df


def make_predictions(pipeline_data: dict, df: pd.DataFrame) -> tuple:
    """Make predictions using the pipeline"""
    try:
        pipeline = pipeline_data['pipeline']
        feature_columns = pipeline_data['feature_columns']
        
        # Select features for prediction
        X = df[feature_columns].copy()
        
        # Ensure categorical columns are strings and numeric columns are numeric
        categorical_features = ['landcover', 'zone_number']  # Default categorical features
        
        for col in X.columns:
            if col in categorical_features:
                # Convert categorical to string
                X[col] = X[col].astype(str)
            else:
                # Convert numeric columns to float, handling any non-numeric values
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill any remaining NaN values that might have been created during conversion
        X = X.fillna(0)
        
        # Make predictions
        predictions = pipeline.predict(X)
        
        # Calculate confidence
        confidence = calculate_pipeline_confidence(pipeline, X)
        
        return predictions, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.exception(e)
        st.stop()


def display_prediction_results(df: pd.DataFrame, predictions: np.ndarray, 
                             target_column: str, confidence: float):
    """Display prediction results with visualizations"""
    
    predicted_column = f"{target_column}_predicted"
    df[predicted_column] = predictions
    
    # Results summary
    st.subheader("Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(predictions))
    with col2:
        st.metric("Mean Predicted Value", f"{np.mean(predictions):.4f}")
    with col3:
        st.metric("Model Confidence", f"{confidence:.2f}")
    
    # Show detailed results
    st.dataframe(df, use_container_width=True)
    
    # Performance comparison if ground truth available
    if target_column in df.columns:
        st.subheader("Model Performance on Test Data")
        
        actual_values = df[target_column]
        
        # Calculate metrics
        mse = np.mean((actual_values - predictions) ** 2)
        mae = np.mean(np.abs(actual_values - predictions))
        r2 = np.corrcoef(actual_values, predictions)[0, 1] ** 2
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Squared Error", f"{mse:.4f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col3:
            st.metric("R² Score", f"{r2:.4f}")
        
        # Scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(actual_values, predictions, alpha=0.7, edgecolors='k')
        ax.plot([actual_values.min(), actual_values.max()], 
                [actual_values.min(), actual_values.max()], 'r--')
        ax.set_xlabel(f'Actual {target_column}')
        ax.set_ylabel(f'Predicted {target_column}')
        ax.set_title('Predictions vs Actual Values')
        st.pyplot(fig)
        plt.close()
    
    return df


def create_spatial_visualization(df: pd.DataFrame, predicted_column: str):
    """Create spatial visualization of predictions"""
    
    required_cols = ['latitude', 'longitude']
    if not all(col in df.columns for col in required_cols):
        st.warning("Latitude and longitude columns required for spatial visualization")
        return
    
    st.subheader("Spatial Visualization")
    
    # Aggregation options
    col1, col2 = st.columns(2)
    with col1:
        aggregate_data = st.checkbox("Aggregate by location", value=True,
                                   help="Average predictions for same coordinates")
    
    visualization_df = df.copy()
    
    if aggregate_data:
        # Aggregate predictions by coordinates
        agg_df = visualization_df.groupby(['latitude', 'longitude'])[predicted_column].agg([
            'mean', 'count', 'std'
        ]).reset_index()
        agg_df.columns = ['latitude', 'longitude', f'{predicted_column}_mean', 'count', 'std']
        visualization_df = agg_df
        predicted_column = f'{predicted_column}_mean'
    
    with col2:
        if not aggregate_data and 'upper_depth' in df.columns and 'lower_depth' in df.columns:
            # Depth filtering for non-aggregated data
            upper_depth = st.number_input("Upper depth (cm)", min_value=0, max_value=100, value=0)
            lower_depth = st.number_input("Lower depth (cm)", min_value=0, max_value=100, value=30)
            
            visualization_df = visualization_df[
                (visualization_df['upper_depth'] >= upper_depth) & 
                (visualization_df['lower_depth'] <= lower_depth)
            ]
    
    if len(visualization_df) == 0:
        st.warning("No data available for visualization with current filters")
        return
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Heatmap
        st.write("**Prediction Heatmap**")
        try:
            heatmap = folium_map(visualization_df, predicted_column)
            st.components.v1.html(heatmap._repr_html_(), height=400)
        except Exception as e:
            st.error(f"Error creating heatmap: {str(e)}")
    
    with col2:
        # Point map
        st.write("**Individual Predictions**")
        try:
            center_lat = visualization_df['latitude'].mean()
            center_lon = visualization_df['longitude'].mean()
            
            point_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            
            for _, row in visualization_df.head(1000).iterrows():  # Limit for performance
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    color='red',
                    fill=True,
                    fillColor='blue',
                    fillOpacity=0.6,
                    popup=f"{predicted_column}: {row[predicted_column]:.3f}"
                ).add_to(point_map)
            
            st.components.v1.html(point_map._repr_html_(), height=400)
        except Exception as e:
            st.error(f"Error creating point map: {str(e)}")


def create_download_options(df: pd.DataFrame, pipeline_metadata: dict):
    """Create download options for results"""
    
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions (CSV)",
            data=csv,
            file_name=f"predictions_{pipeline_metadata['title']}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary report
        summary = {
            "model_info": {
                "title": pipeline_metadata['title'],
                "model_type": pipeline_metadata['model_type'],
                "target": pipeline_metadata['target']
            },
            "prediction_stats": {
                "count": len(df),
                "mean": float(df.iloc[:, -1].mean()) if len(df) > 0 else 0,
                "std": float(df.iloc[:, -1].std()) if len(df) > 0 else 0,
                "min": float(df.iloc[:, -1].min()) if len(df) > 0 else 0,
                "max": float(df.iloc[:, -1].max()) if len(df) > 0 else 0
            }
        }
        
        st.download_button(
            label="Download Summary (JSON)",
            data=json.dumps(summary, indent=2),
            file_name=f"prediction_summary_{pipeline_metadata['title']}.json",
            mime="application/json"
        )


def show():
    """Main prediction application"""
    
    st.header("Enhanced Prediction System")
    
    with st.expander("About Pipeline Predictions"):
        st.markdown("""
        ### Enhanced Prediction Features
        
        This updated prediction system uses **scikit-learn pipelines** for:
        - **Consistent preprocessing**: Same transformations as training
        - **Simplified workflow**: Single pipeline handles all steps
        - **Better reliability**: Reduced chance of preprocessing errors
        - **Enhanced visualization**: Improved spatial mapping and analysis
        
        **Key Improvements:**
        - Pipeline-based architecture for consistency
        - Better error handling and validation
        - Enhanced spatial visualization options  
        - Comprehensive result analysis
        - Multiple download formats
        """)
    
    # Model selection
    st.subheader("Select Trained Model")
    selected_pipeline = create_pipeline_selector()
    
    # Load pipeline
    with st.spinner("Loading pipeline..."):
        pipeline_data = load_pipeline_model(selected_pipeline['pipeline_path'])
        
        st.success(f"Loaded: {selected_pipeline['title']} ({selected_pipeline['model_type']})")
        
        # Display model info
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Target Variable:** {selected_pipeline['target']}")
            feature_count = len(json.loads(selected_pipeline['feature_columns']))
            st.write(f"**Features:** {feature_count}")
        with col2:
            if selected_pipeline.get('metrics'):
                metrics = json.loads(selected_pipeline['metrics'])
                st.write(f"**Training MAE:** {metrics.get('mae', 'N/A'):.4f}")
                st.write(f"**Training R²:** {metrics.get('r2', 'N/A'):.4f}")
    
    # Template generation
    st.subheader("Data Requirements")
    
    with st.spinner("Generating template..."):
        template_df = generate_template_dataset(selected_pipeline)
        
        st.write("**Required columns for prediction:**")
        st.write(", ".join(template_df.columns))
        
        # Template download
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="Download Template Dataset",
            data=template_csv,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="Download a template with required column structure"
        )
    
    # File upload
    st.subheader("Upload Prediction Data")
    uploaded_file = st.file_uploader(
        "Choose prediction dataset",
        type=["csv", "xlsx"],
        help="Upload data following the template structure"
    )
    
    if uploaded_file:
        try:
            # Load data
            with st.spinner("Loading prediction dataset..."):
                if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                    prediction_df = pd.read_excel(uploaded_file)
                else:
                    prediction_df = pd.read_csv(uploaded_file)
                
                st.success(f"Loaded {len(prediction_df):,} rows for prediction")
                
                # Preview data
                with st.expander("Preview Dataset"):
                    st.dataframe(prediction_df.head(), use_container_width=True)
            
            # Validate data
            st.subheader("Data Validation")
            validated_df = validate_prediction_data(prediction_df, pipeline_data)
            
            # Make predictions
            st.subheader("Making Predictions")
            with st.spinner("Generating predictions..."):
                predictions, confidence = make_predictions(pipeline_data, validated_df)
                
                st.success(f"Generated {len(predictions):,} predictions")
            
            # Display results
            results_df = display_prediction_results(
                validated_df, predictions, 
                selected_pipeline['target'], confidence
            )
            
            # Spatial visualization
            if 'latitude' in results_df.columns and 'longitude' in results_df.columns:
                create_spatial_visualization(
                    results_df, f"{selected_pipeline['target']}_predicted"
                )
            
            # Download options
            create_download_options(results_df, selected_pipeline)
            
        except Exception as e:
            st.error(f"Error processing predictions: {str(e)}")
            st.exception(e)
    
    else:
        st.info("Upload a prediction dataset to begin")


if __name__ == "__main__":
    sidebar(title="Enhanced Predictions")
    show()