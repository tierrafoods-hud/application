import streamlit as st
from utils.sidebar_menu import sidebar
from utils.pipeline_manager import load_pipelines_from_db
from config.database import get_db
from dotenv import load_dotenv
import os
import json
import pandas as pd
from datetime import datetime

load_dotenv()

# Database setup
db_type = os.getenv('DB_TYPE')
if not db_type:
    st.error("DB_TYPE environment variable not set")
    st.stop()

DB = get_db(db_type)


def format_metrics_display(metrics_data):
    """Format metrics for display"""
    if isinstance(metrics_data, str):
        try:
            metrics = json.loads(metrics_data)
        except:
            return None
    else:
        metrics = metrics_data
    
    return metrics


def format_feature_display(feature_data):
    """Format features for display"""
    if isinstance(feature_data, str):
        try:
            features = json.loads(feature_data)
        except:
            return []
    else:
        features = feature_data
    
    return features if isinstance(features, list) else []


def create_model_card(model):
    """Create an enhanced model card display"""
    
    # Check file existence
    file_exists = os.path.exists(model['pipeline_path'])
    
    # Status indicator
    status_color = "🟢" if file_exists and model.get('is_active', True) else "🔴"
    
    with st.container(border=True):
        # Header with status
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"### {status_color} {model['title']}")
        with col2:
            if model.get('version'):
                st.markdown(f"**Version:** `v{model['version']}`")
        
        # Main info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Model Type:** {model['model_type']}")
            st.markdown(f"**Target:** {model['target']}")
            
            # Feature information
            feature_columns = format_feature_display(model['feature_columns'])
            categorical_features = format_feature_display(model.get('categorical_features', []))
            numeric_features = format_feature_display(model.get('numeric_features', []))
            
            st.markdown(f"**Total Features:** {len(feature_columns)}")
            if categorical_features:
                st.markdown(f"**Categorical:** {len(categorical_features)}")
            if numeric_features:
                st.markdown(f"**Numeric:** {len(numeric_features)}")
        
        with col2:
            # Timestamps
            if model.get('created_at'):
                st.markdown(f"**Created:** {model['created_at']}")
            if model.get('updated_at'):
                st.markdown(f"**Updated:** {model['updated_at']}")
            
            # File info
            if file_exists:
                file_size = os.path.getsize(model['pipeline_path'])
                st.markdown(f"**File Size:** {file_size / 1024 / 1024:.2f} MB")
            else:
                st.error("⚠️ Pipeline file missing")
        
        # Performance metrics
        metrics = format_metrics_display(model.get('metrics'))
        if metrics:
            st.markdown("#### Performance Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mse_val = metrics.get('mse', 0)
                st.metric("MSE", f"{mse_val:.4f}")
            with col2:
                mae_val = metrics.get('mae', 0)
                st.metric("MAE", f"{mae_val:.4f}")
            with col3:
                r2_val = metrics.get('r2', 0)
                st.metric("R²", f"{r2_val:.4f}")
        
        # Description
        if model.get('description'):
            st.markdown("#### Description")
            st.markdown(model['description'])
        
        # Feature details in expander
        if feature_columns:
            with st.expander("Feature Details"):
                if categorical_features:
                    st.markdown("**Categorical Features:**")
                    st.write(", ".join(categorical_features))
                
                if numeric_features:
                    st.markdown("**Numeric Features:**")
                    st.write(", ".join(numeric_features))
        
        # Training configuration
        training_config = model.get('training_config')
        if training_config:
            try:
                if isinstance(training_config, str):
                    config = json.loads(training_config)
                else:
                    config = training_config
                
                with st.expander("Training Configuration"):
                    for key, value in config.items():
                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            except:
                pass
        
        # Actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if file_exists:
                with open(model['pipeline_path'], 'rb') as f:
                    st.download_button(
                        label="Download Pipeline",
                        data=f,
                        file_name=f"{model['title']}_pipeline.pkl",
                        key=f"download_{model['id']}",
                        type="primary"
                    )
            else:
                st.button(
                    "File Missing", 
                    disabled=True,
                    key=f"missing_{model['id']}"
                )
        
        with col2:
            if st.button("Use for Prediction", key=f"predict_{model['id']}"):
                st.switch_page("pages/make_predictions.py")
        
        with col3:
            # Deactivate/Activate toggle
            if model.get('is_active', True):
                if st.button("Deactivate", key=f"deactivate_{model['id']}", type="secondary"):
                    update_model_status(model['id'], False)
                    st.rerun()
            else:
                if st.button("Activate", key=f"activate_{model['id']}", type="secondary"):
                    update_model_status(model['id'], True)
                    st.rerun()


def update_model_status(model_id: int, is_active: bool):
    """Update model active status"""
    try:
        query = "UPDATE models SET is_active = %s WHERE id = %s"
        DB.update(query, (is_active, model_id))
        st.success(f"Model {'activated' if is_active else 'deactivated'} successfully")
    except Exception as e:
        st.error(f"Error updating model status: {str(e)}")


def create_models_summary(models):
    """Create summary statistics"""
    if not models:
        return
    
    st.subheader("Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_models = len(models)
        st.metric("Total Models", total_models)
    
    with col2:
        active_models = sum(1 for m in models if m.get('is_active', True))
        st.metric("Active Models", active_models)
    
    with col3:
        model_types = set(m['model_type'] for m in models)
        st.metric("Model Types", len(model_types))
    
    with col4:
        targets = set(m['target'] for m in models)
        st.metric("Target Variables", len(targets))
    
    # Model type distribution
    if len(models) > 1:
        st.subheader("Model Distribution")
        
        # Create DataFrame for visualization
        df_summary = pd.DataFrame(models)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**By Model Type:**")
            type_counts = df_summary['model_type'].value_counts()
            st.bar_chart(type_counts)
        
        with col2:
            st.markdown("**By Target Variable:**")
            target_counts = df_summary['target'].value_counts()
            st.bar_chart(target_counts)


def show():
    """Main display function"""
    
    st.header("Pipeline Model Registry")
    
    with st.expander("About Model Management"):
        st.markdown("""
        ### Enhanced Model Registry
        
        This registry shows all trained pipeline models with:
        - **Pipeline Architecture**: Complete scikit-learn pipelines
        - **Enhanced Metadata**: Detailed feature and configuration info
        - **Status Management**: Activate/deactivate models
        - **Performance Tracking**: Training metrics and file information
        - **Quick Actions**: Download, predict, and manage models
        
        **Status Indicators:**
        - 🟢 Active and available
        - 🔴 Inactive or file missing
        """)
    
    # Filter options
    st.subheader("Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_inactive = st.checkbox("Show Inactive Models", value=True)
    
    with col2:
        model_type_filter = st.selectbox(
            "Filter by Model Type",
            options=["All"] + ["Random Forest Regressor", "Gradient Boosting Regressor", 
                     "Linear Regression", "Support Vector Regressor", 
                     "K-Nearest Neighbors Regressor", "MLP Regressor (neural network)"],
            index=0
        )
    
    with col3:
        target_filter = st.selectbox(
            "Filter by Target",
            options=["All", "orgc", "tceq"],
            index=0
        )
    
    # Load models
    with st.spinner("Loading pipeline models..."):
        try:
            models = load_pipelines_from_db(DB)
            
            if not models:
                st.warning("No pipeline models found. Train a model first using the 'Enhanced Model Training' page.")
                return
            
            # Apply filters
            filtered_models = models
            
            if not show_inactive:
                filtered_models = [m for m in filtered_models if m.get('is_active', True)]
            
            if model_type_filter != "All":
                filtered_models = [m for m in filtered_models if m['model_type'] == model_type_filter]
            
            if target_filter != "All":
                filtered_models = [m for m in filtered_models if m['target'] == target_filter]
            
            if not filtered_models:
                st.warning("No models match the current filters.")
                return
            
            # Show summary
            create_models_summary(filtered_models)
            
            st.subheader(f"Models ({len(filtered_models)} found)")
            
            # Display models
            for model in filtered_models:
                create_model_card(model)
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    sidebar(
        title="Model Registry",
        about="Enhanced model registry for pipeline-based ML models. View, manage, and download trained models with comprehensive metadata and performance tracking."
    )
    
    show()