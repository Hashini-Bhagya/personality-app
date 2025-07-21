import joblib
import pandas as pd
import numpy as np
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning
from config import Config

# Suppress version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

def patch_ordinal_encoder(encoder):
    """Patch OrdinalEncoder to add missing attributes"""
    if not hasattr(encoder, '_infrequent_enabled'):
        encoder._infrequent_enabled = False
    if not hasattr(encoder, 'n_features_in_'):
        encoder.n_features_in_ = len(encoder.categories_)
    return encoder

def patch_preprocessor(preprocessor):
    """Recursively patch all OrdinalEncoders in the preprocessor"""
    for i, (name, transformer, columns) in enumerate(preprocessor.transformers_):
        if hasattr(transformer, 'steps'):
            # Pipeline object - look for encoder step
            for j, (step_name, step_transformer) in enumerate(transformer.steps):
                if 'encoder' in step_name.lower() and hasattr(step_transformer, 'categories_'):
                    # Patch the encoder
                    patched_encoder = patch_ordinal_encoder(step_transformer)
                    transformer.steps[j] = (step_name, patched_encoder)
        elif hasattr(transformer, 'categories_'):
            # Direct OrdinalEncoder - patch it
            preprocessor.transformers_[i] = (name, patch_ordinal_encoder(transformer), columns)
    return preprocessor

def load_compatible_model(path):
    """Load model with cross-version compatibility"""
    try:
        model = joblib.load(path)
        
        # XGBoost specific patches for version 2.0.3
        try:
            from xgboost import XGBClassifier
            if isinstance(model, XGBClassifier):
                # Remove deprecated attributes
                for attr in ['use_label_encoder', 'validate_parameters']:
                    if hasattr(model, attr):
                        delattr(model, attr)
                
                # Add missing methods if needed
                if not hasattr(model, 'get_booster'):
                    model.get_booster = lambda: model._Booster
                
                # Handle feature count
                if not hasattr(model, 'n_features_in_'):
                    try:
                        model.n_features_in_ = model._le.n_features_in_
                    except AttributeError:
                        try:
                            model.n_features_in_ = model.get_booster().num_features()
                        except:
                            # Last resort fallback
                            model.n_features_in_ = 0
        except ImportError:
            pass  # XGBoost not installed
        
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

class PersonalityClassifier:
    def __init__(self):
        try:
            self.model = load_compatible_model(Config.MODEL_PATH)
            self.preprocessor = joblib.load(Config.PREPROCESSOR_PATH)
            
            # Patch preprocessor for version compatibility
            self.preprocessor = patch_preprocessor(self.preprocessor)
            
            # List of expected engineered features
            self.expected_features = [
                'Social_Going_ratio', 'Social_Going_diff',
                'Friend_Post_ratio', 'Friend_Post_product',
                'Avoids_Interaction',
                'Social_Index', 'Social_Std',
                'Alone_Ratio', 'Alone_Log'
            ]
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
        
    def clean_column_names(self, df):
        """Clean column names to match training data format"""
        df.columns = [re.sub(r'\W+', '_', col.strip()) for col in df.columns]
        return df
        
    def create_features(self, df):
        """Feature engineering pipeline with fallback for missing features"""
        df = df.copy()
        
        # Ensure all base columns exist with default values
        base_columns = [
            'Time_spent_Alone',
            'Social_event_attendance',
            'Going_outside',
            'Friends_circle_size',
            'Post_frequency',
            'Stage_fear',
            'Drained_after_socializing'
        ]
        
        for col in base_columns:
            if col not in df.columns:
                if col in ['Stage_fear', 'Drained_after_socializing']:
                    df[col] = 'No'  # Default for categorical
                else:
                    df[col] = 0      # Default for numerical
        
        # Social interaction features
        if 'Going_outside' in df.columns and 'Social_event_attendance' in df.columns:
            df['Social_Going_ratio'] = df['Social_event_attendance'] / df['Going_outside'].replace(0, 0.1)
            df['Social_Going_diff'] = df['Social_event_attendance'] - df['Going_outside']
        else:
            df['Social_Going_ratio'] = 0
            df['Social_Going_diff'] = 0
        
        # Friend-post features
        if 'Friends_circle_size' in df.columns and 'Post_frequency' in df.columns:
            df['Friend_Post_ratio'] = df['Friends_circle_size'] / (df['Post_frequency'].replace(0, 0.1) + 1e-5)
            df['Friend_Post_product'] = df['Friends_circle_size'] * df['Post_frequency']
        else:
            df['Friend_Post_ratio'] = 0
            df['Friend_Post_product'] = 0
        
        # Psychological features
        if 'Stage_fear' in df.columns and 'Drained_after_socializing' in df.columns:
            df['Avoids_Interaction'] = np.where(
                (df['Stage_fear'] == 'Yes') | (df['Drained_after_socializing'] == 'Yes'), 1, 0
            )
        else:
            df['Avoids_Interaction'] = 0
        
        # Social indices
        social_cols = [c for c in ['Social_event_attendance', 'Going_outside', 
                                  'Friends_circle_size', 'Post_frequency'] if c in df.columns]
        if social_cols:
            df['Social_Index'] = df[social_cols].mean(axis=1)
            df['Social_Std'] = df[social_cols].std(axis=1).fillna(0)
        else:
            df['Social_Index'] = 0
            df['Social_Std'] = 0
        
        # Alone time features
        if 'Time_spent_Alone' in df.columns:
            df['Alone_Ratio'] = df['Time_spent_Alone'] / 24
            df['Alone_Log'] = np.log1p(df['Time_spent_Alone'])
        else:
            df['Alone_Ratio'] = 0
            df['Alone_Log'] = 0
        
        # Ensure all expected features exist
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
                
        return df
        
    def predict(self, input_data):
        """Predict personality from input data"""
        try:
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Clean column names
            input_df = self.clean_column_names(input_df)
            
            # Feature engineering
            input_df = self.create_features(input_df)
            
            # Preprocess
            processed_data = self.preprocessor.transform(input_df)
            
            # Predict
            prediction = self.model.predict(processed_data)[0]
            personality = 'Introvert' if prediction == 0 else 'Extrovert'
            
            # Confidence score
            probabilities = self.model.predict_proba(processed_data)[0]
            confidence = probabilities[0] if prediction == 0 else probabilities[1]
            
            return personality, float(confidence)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")