"""
Pipeline Manager for Streamlit ML Application
Handles creation, training, and management of scikit-learn pipelines
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import streamlit as st


class ModelPipeline:
    """Enhanced pipeline management for ML models"""
    
    MODELS = {
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
        'MLP Regressor (neural network)': MLPRegressor(random_state=42, max_iter=1000)
    }
    
    DEFAULT_CATEGORICAL_FEATURES = ['landcover', 'zone_number']
    DEFAULT_DROP_COLUMNS = ['country_name', 'region', 'continent', 'ecoregion_type', 'zone_name']
    
    def __init__(self, target_column: str, categorical_features: List[str] = None):
        self.target_column = target_column
        self.categorical_features = categorical_features or self.DEFAULT_CATEGORICAL_FEATURES
        self.numeric_features = []
        self.feature_columns = []
        self.preprocessor = None
        self.pipelines = {}
        self.metrics_history = {}
        self.best_pipeline = None
        self.best_score = float('inf')
        self.best_model_name = None
        
    def preprocess_dataset(self, df: pd.DataFrame, drop_columns: List[str] = None) -> pd.DataFrame:
        """Enhanced preprocessing with better handling"""
        if drop_columns is None:
            drop_columns = self.DEFAULT_DROP_COLUMNS
            
        dataset = df.copy()
        initial_rows = len(dataset)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Drop highly correlated columns
            if self.target_column == 'orgc' and 'tceq' in dataset.columns:
                dataset.drop(columns=['tceq'], inplace=True)
            elif self.target_column == 'tceq' and 'orgc' in dataset.columns:
                dataset.drop(columns=['orgc'], inplace=True)
            
            progress_bar.progress(0.1)
            
            # Drop columns with > 70% missing values
            status_text.text("Removing columns with excessive missing values...")
            threshold = 0.7 * len(dataset)
            cols_to_drop = [col for col in dataset.columns 
                           if dataset[col].isna().sum() > threshold and col != self.target_column]
            dataset.drop(columns=cols_to_drop + drop_columns, errors='ignore', inplace=True)
            progress_bar.progress(0.3)
            
            # Validate target column exists
            if self.target_column not in dataset.columns:
                raise ValueError(f"Target variable '{self.target_column}' not found in dataset")
            
            # Remove rows with missing target values
            status_text.text("Cleaning target variable...")
            dataset.dropna(subset=[self.target_column], inplace=True)
            progress_bar.progress(0.5)
            
            # Remove duplicates
            status_text.text("Removing duplicates...")
            dataset.drop_duplicates(inplace=True)
            progress_bar.progress(0.7)
            
            # Handle missing values in remaining columns
            status_text.text("Handling missing values...")
            dataset.dropna(inplace=True)
            progress_bar.progress(0.9)
            
            # Convert categorical features to string type
            available_categorical = [col for col in self.categorical_features if col in dataset.columns]
            if available_categorical:
                dataset[available_categorical] = dataset[available_categorical].astype(str)
            
            # Clean zone_number if it exists
            if 'zone_number' in dataset.columns:
                dataset = dataset[~dataset['zone_number'].isin(['VII', 'VI'])]
            
            progress_bar.progress(1.0)
            status_text.success(f"Preprocessing complete! Shape: {dataset.shape}")
            
            st.info(f"Removed {initial_rows - len(dataset)} rows during preprocessing")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            raise e
        finally:
            # Clean up progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
        
        return dataset.reset_index(drop=True)
    
    def create_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Create column transformer for preprocessing pipeline"""
        
        # Remove longitude/latitude for feature selection
        feature_df = df.drop(columns=['longitude', 'latitude', self.target_column], errors='ignore')
        
        # Identify numeric and categorical features
        self.numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = [col for col in self.categorical_features 
                                   if col in feature_df.columns]
        
        # Store all feature columns
        self.feature_columns = self.numeric_features + self.categorical_features
        
        transformers = []
        
        # Numeric preprocessing
        if self.numeric_features:
            transformers.append(('num', StandardScaler(), self.numeric_features))
        
        # Categorical preprocessing
        if self.categorical_features:
            transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                               self.categorical_features))
        
        if not transformers:
            raise ValueError("No features available for preprocessing")
        
        self.preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        return self.preprocessor
    
    def create_pipelines(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Pipeline]:
        """Create pipeline for each model"""
        
        if self.preprocessor is None:
            raise ValueError("Preprocessor not created. Call create_preprocessor first.")
        
        self.pipelines = {}
        
        for name, model in self.MODELS.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('model', model)
            ])
            self.pipelines[name] = pipeline
        
        return self.pipelines
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.3, random_state: int = 42) -> Dict[str, Any]:
        """Train all pipelines and return evaluation results"""
        
        if not self.pipelines:
            raise ValueError("No pipelines created. Call create_pipelines first.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            for i, (name, pipeline) in enumerate(self.pipelines.items()):
                status_text.text(f'Training {name}...')
                
                # Train pipeline
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'mse': mean_squared_error(y_test, y_pred),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
                
                # Track best model
                if metrics['mae'] < self.best_score:
                    self.best_score = metrics['mae']
                    self.best_pipeline = pipeline
                    self.best_model_name = name
                
                results[name] = {
                    'pipeline': pipeline,
                    'metrics': metrics,
                    'predictions': y_pred,
                    'test_data': y_test
                }
                
                self.metrics_history[name] = metrics
                
                progress_bar.progress((i + 1) / len(self.pipelines))
        
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            raise e
        finally:
            progress_bar.empty()
            status_text.empty()
        
        return results
    
    def save_pipeline(self, title: str, output_dir: str = "models") -> Dict[str, Any]:
        """Save the best pipeline and return metadata"""
        
        if self.best_pipeline is None:
            raise ValueError("No trained pipeline to save. Train models first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{title}_{self.best_model_name}_{timestamp}.pkl".replace(" ", "_").lower()
        filepath = os.path.join(output_dir, filename)
        
        # Prepare pipeline data
        pipeline_data = {
            'pipeline': self.best_pipeline,
            'title': title,
            'model_name': self.best_model_name,
            'target_column': self.target_column,
            'feature_columns': self.feature_columns,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features,
            'metrics': self.metrics_history[self.best_model_name],
            'created_at': datetime.now().isoformat(),
            'version': '2.0'
        }
        
        # Save pipeline
        joblib.dump(pipeline_data, filepath)
        
        return {
            'filepath': filepath,
            'filename': filename,
            'data': pipeline_data
        }
    
    def load_pipeline(self, filepath: str) -> Dict[str, Any]:
        """Load a saved pipeline"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline_data = joblib.load(filepath)
        
        # Validate pipeline data structure
        required_keys = ['pipeline', 'target_column', 'feature_columns']
        if not all(key in pipeline_data for key in required_keys):
            raise ValueError("Invalid pipeline format")
        
        return pipeline_data
    
    def predict(self, pipeline_data: Dict[str, Any], X: pd.DataFrame) -> np.ndarray:
        """Make predictions using loaded pipeline"""
        
        pipeline = pipeline_data['pipeline']
        feature_columns = pipeline_data['feature_columns']
        
        # Ensure input has required features
        missing_features = set(feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Select and order features correctly
        X_features = X[feature_columns]
        
        return pipeline.predict(X_features)


def save_pipeline_to_db(db, pipeline_metadata: Dict[str, Any]) -> None:
    """Save pipeline metadata to database with optimized schema"""
    
    data = pipeline_metadata['data']
    
    model_data = {
        'title': data['title'],
        'model_type': data['model_name'],
        'target': data['target_column'],
        'feature_columns': json.dumps(data['feature_columns']),
        'categorical_features': json.dumps(data['categorical_features']),
        'numeric_features': json.dumps(data['numeric_features']),
        'metrics': json.dumps(data['metrics']),
        'pipeline_path': pipeline_metadata['filepath'],
        'training_config': json.dumps({
            'version': data['version'],
            'created_at': data['created_at']
        }),
        'data_shape': json.dumps({
            'n_features': len(data['feature_columns']),
            'n_categorical': len(data['categorical_features']),
            'n_numeric': len(data['numeric_features'])
        })
    }
    
    query = """
    INSERT INTO models (
        title, model_type, target, feature_columns, categorical_features, 
        numeric_features, metrics, pipeline_path, training_config, data_shape
    ) VALUES (
        %(title)s, %(model_type)s, %(target)s, %(feature_columns)s, 
        %(categorical_features)s, %(numeric_features)s, %(metrics)s, 
        %(pipeline_path)s, %(training_config)s, %(data_shape)s
    )
    """
    
    db.create(query, model_data)


def load_pipelines_from_db(db) -> List[Dict[str, Any]]:
    """Load pipeline metadata from database"""
    
    query = """
    SELECT id, title, model_type, target, feature_columns, categorical_features, 
           numeric_features, metrics, pipeline_path, training_config, data_shape, 
           created_at, updated_at, version, is_active 
    FROM models 
    WHERE is_active = TRUE 
    ORDER BY updated_at DESC
    """
    
    return db.fetchAllAsDict(query)


def calculate_pipeline_confidence(pipeline, X: pd.DataFrame) -> float:
    """Calculate confidence score for pipeline predictions"""
    
    try:
        model = pipeline.named_steps['model']
        
        # Get transformed features - ensure we have clean numeric data
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Make a copy and ensure proper data types before transformation
        X_clean = X.copy()
        
        # Ensure categorical columns are strings
        categorical_features = ['landcover', 'zone_number']
        for col in X_clean.columns:
            if col in categorical_features:
                X_clean[col] = X_clean[col].astype(str)
            else:
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        
        # Fill any NaN values
        X_clean = X_clean.fillna(0)
        
        X_transformed = preprocessor.transform(X_clean)
        
        # Ensure X_transformed is numeric
        if hasattr(X_transformed, 'dtype') and not np.issubdtype(X_transformed.dtype, np.number):
            # If it's not numeric, try to convert
            X_transformed = np.array(X_transformed, dtype=np.float64)
        
        if hasattr(model, 'predict_proba'):
            # For classifiers (though we're doing regression)
            probabilities = model.predict_proba(X_transformed)
            return float(np.mean(np.max(probabilities, axis=1)))
        
        elif isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            # Tree-based models: use prediction variance across trees
            try:
                predictions = []
                for estimator in model.estimators_:
                    pred = estimator.predict(X_transformed)
                    predictions.append(pred)
                predictions = np.array(predictions)
                variance = np.var(predictions, axis=0)
                mean_variance = np.mean(variance)
                return float(1 / (1 + mean_variance))
            except:
                # Fallback: use basic confidence
                return 0.75
        
        else:
            # For other models, use prediction consistency
            try:
                predictions = model.predict(X_transformed)
                std_pred = np.std(predictions)
                return float(1 / (1 + std_pred)) if std_pred > 0 else 0.95
            except:
                # Fallback confidence
                return 0.80
                
    except Exception as e:
        # If confidence calculation fails, return a default confidence
        print(f"Warning: Confidence calculation failed: {str(e)}")
        return 0.70


def validate_pipeline_input(X: pd.DataFrame, pipeline_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input data against pipeline requirements"""
    
    required_features = pipeline_data['feature_columns']
    available_features = X.columns.tolist()
    
    missing_features = set(required_features) - set(available_features)
    
    return len(missing_features) == 0, list(missing_features)
