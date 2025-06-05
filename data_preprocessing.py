import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Data preprocessing pipeline for diabetes dataset"""
    
    def __init__(self, scaling_method='standard', handle_missing='median'):
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        
        # Initialize preprocessors
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        
        # Fitted flag
        self.is_fitted = False
        
        # Feature names
        self.feature_columns = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.target_column = 'Outcome'
    
    def fit_transform(self, data):
        """Fit preprocessors and transform data"""
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values (zeros in this dataset often represent missing values)
        df = self._handle_missing_values(df)
        
        # Separate features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Fit and transform features
        X_processed = self._fit_transform_features(X)
        
        # Process target (already binary, but ensure correct format)
        y_processed = self._process_target(y)
        
        self.is_fitted = True
        
        return X_processed, y_processed
    
    def transform(self, data):
        """Transform new data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Handle different input formats
        if isinstance(data, pd.DataFrame):
            X = data[self.feature_columns].values
        elif isinstance(data, np.ndarray):
            if data.shape[1] == len(self.feature_columns):
                X = data
            else:
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {data.shape[1]}")
        else:
            raise ValueError("Data must be pandas DataFrame or numpy array")
        
        # Apply transformations
        if self.imputer:
            X = self.imputer.transform(X)
        
        if self.scaler:
            X = self.scaler.transform(X)
        
        return X
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_processed = df.copy()
        
        # In diabetes dataset, 0 values in certain columns represent missing data
        missing_value_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        for col in missing_value_columns:
            if col in df_processed.columns:
                # Replace 0 with NaN for proper imputation
                df_processed[col] = df_processed[col].replace(0, np.nan)
        
        return df_processed
    
    def _fit_transform_features(self, X):
        """Fit and transform features"""
        # Handle missing values
        if self.handle_missing == 'median':
            self.imputer = SimpleImputer(strategy='median')
        elif self.handle_missing == 'mean':
            self.imputer = SimpleImputer(strategy='mean')
        else:
            self.imputer = SimpleImputer(strategy='most_frequent')
        
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
        
        if self.scaler:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        return X_scaled
    
    def _process_target(self, y):
        """Process target variable"""
        # Ensure target is binary (0, 1)
        y_processed = y.astype(int)
        
        # Verify binary classification
        unique_values = np.unique(y_processed)
        if not np.array_equal(unique_values, [0, 1]) and not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [1]):
            # If not binary, use label encoder
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y_processed)
        
        return y_processed
    
    def get_feature_names(self):
        """Get feature names"""
        return self.feature_columns
    
    def get_preprocessing_info(self):
        """Get information about preprocessing steps"""
        info = {
            'scaling_method': self.scaling_method,
            'missing_value_strategy': self.handle_missing,
            'is_fitted': self.is_fitted,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        if self.scaler:
            info['scaler_type'] = type(self.scaler).__name__
            if hasattr(self.scaler, 'mean_'):
                info['feature_means'] = self.scaler.mean_.tolist()
            if hasattr(self.scaler, 'scale_'):
                info['feature_scales'] = self.scaler.scale_.tolist()
        
        if self.imputer:
            info['imputer_strategy'] = self.imputer.strategy
            if hasattr(self.imputer, 'statistics_'):
                info['imputation_values'] = self.imputer.statistics_.tolist()
        
        return info
    
    def create_data_splits(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """Create train/validation/test splits"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: separate train and validation
        if val_size > 0:
            val_size_adjusted = val_size / (1 - test_size)  # Adjust for already removed test set
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
            )
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        else:
            return {
                'X_train': X_temp, 'y_train': y_temp,
                'X_test': X_test, 'y_test': y_test
            }
    
    def get_data_statistics(self, X, y):
        """Get statistics about the data"""
        stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': np.bincount(y.astype(int)),
            'feature_stats': {}
        }
        
        for i, feature_name in enumerate(self.feature_columns[:X.shape[1]]):
            feature_data = X[:, i]
            stats['feature_stats'][feature_name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'median': np.median(feature_data)
            }
        
        return stats
