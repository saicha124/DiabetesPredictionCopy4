import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import copy
import time

class ClientSimulator:
    """Simulates a federated learning client"""
    
    def __init__(self, client_id, data, model_type='logistic_regression'):
        self.client_id = client_id
        self.data = data
        self.model_type = model_type
        self.local_model = None
        self.global_model = None
        self.training_history = []
        
        # Initialize local model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local model"""
        if self.model_type == 'logistic_regression':
            self.local_model = LogisticRegression(
                random_state=42 + self.client_id,
                max_iter=1000,
                solver='liblinear'
            )
        elif self.model_type == 'random_forest':
            self.local_model = RandomForestClassifier(
                n_estimators=10,  # Small for faster training
                random_state=42 + self.client_id,
                max_depth=5
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def receive_global_model(self, global_model):
        """Receive global model parameters from server"""
        self.global_model = copy.deepcopy(global_model)
        
        # Initialize local model with global parameters
        if hasattr(global_model, 'coef_') and hasattr(global_model, 'intercept_'):
            try:
                # For logistic regression, copy parameters
                self.local_model.coef_ = global_model.coef_.copy()
                self.local_model.intercept_ = global_model.intercept_.copy()
                self.local_model.classes_ = global_model.classes_.copy()
            except:
                # If parameter copying fails, proceed with fresh model
                pass
    
    def train(self, local_epochs=1):
        """Train local model and return parameter updates"""
        start_time = time.time()
        
        try:
            X_train = self.data['X_train']
            y_train = self.data['y_train']
            X_test = self.data['X_test']
            y_test = self.data['y_test']
            
            if len(X_train) == 0:
                return self._create_dummy_update()
            
            # Check if we have enough class diversity for logistic regression
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"Client {self.client_id}: Insufficient class diversity, using dummy update")
                return self._create_dummy_update()
            
            # Train local model
            for epoch in range(local_epochs):
                try:
                    self.local_model.fit(X_train, y_train)
                except ValueError as ve:
                    if "class" in str(ve).lower():
                        print(f"Client {self.client_id}: Class-related training error: {ve}")
                        return self._create_dummy_update()
                    else:
                        raise ve
            
            # Evaluate local model
            train_accuracy = test_accuracy = f1 = 0.0
            try:
                if len(X_test) > 0 and len(np.unique(y_test)) > 0:
                    train_predictions = self.local_model.predict(X_train)
                    test_predictions = self.local_model.predict(X_test)
                    
                    train_accuracy = accuracy_score(y_train, train_predictions)
                    test_accuracy = accuracy_score(y_test, test_predictions)
                    if len(np.unique(y_test)) > 1:
                        f1 = f1_score(y_test, test_predictions, average='weighted')
                    else:
                        f1 = 0.0
            except Exception as eval_error:
                print(f"Client {self.client_id} evaluation error: {eval_error}")
            
            # Create parameter update
            update = self._create_parameter_update()
            
            # Record training metrics
            training_time = time.time() - start_time
            metrics = {
                'client_id': self.client_id,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'f1_score': f1,
                'training_time': training_time,
                'samples': len(X_train),
                'classes': len(unique_classes)
            }
            
            self.training_history.append(metrics)
            
            return update
            
        except Exception as e:
            print(f"Client {self.client_id} training error: {e}")
            return self._create_dummy_update()
    
    def _create_parameter_update(self):
        """Create parameter update from local model"""
        try:
            if hasattr(self.local_model, 'coef_') and hasattr(self.local_model, 'intercept_'):
                # For logistic regression
                parameters = np.concatenate([
                    self.local_model.coef_.flatten(),
                    self.local_model.intercept_.flatten()
                ])
            elif hasattr(self.local_model, 'feature_importances_'):
                # For random forest, use feature importances as proxy
                parameters = self.local_model.feature_importances_
            else:
                # Fallback: random parameters
                parameters = np.random.normal(0, 0.1, 10)
            
            update = {
                'client_id': self.client_id,
                'parameters': parameters,
                'num_samples': len(self.data['X_train']),
                'model_type': self.model_type
            }
            
            return update
            
        except Exception as e:
            print(f"Parameter update creation failed for client {self.client_id}: {e}")
            return self._create_dummy_update()
    
    def _create_dummy_update(self):
        """Create a dummy update when training fails"""
        return {
            'client_id': self.client_id,
            'parameters': np.random.normal(0, 0.01, 10),
            'num_samples': 0,
            'model_type': self.model_type
        }
    
    def get_data_statistics(self):
        """Get statistics about client's data"""
        stats = {
            'client_id': self.client_id,
            'train_samples': len(self.data['X_train']),
            'test_samples': len(self.data['X_test']),
            'features': self.data['X_train'].shape[1] if len(self.data['X_train']) > 0 else 0
        }
        
        if len(self.data['y_train']) > 0:
            unique, counts = np.unique(self.data['y_train'], return_counts=True)
            stats['class_distribution'] = dict(zip(unique.astype(int), counts.astype(int)))
        
        return stats
    
    def evaluate(self):
        """Evaluate local model performance"""
        if self.local_model is None:
            return None
        
        try:
            X_test = self.data['X_test']
            y_test = self.data['y_test']
            
            if len(X_test) == 0:
                return None
            
            predictions = self.local_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='weighted')
            
            return {
                'client_id': self.client_id,
                'test_accuracy': accuracy,
                'f1_score': f1,
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            print(f"Client {self.client_id} evaluation error: {e}")
            return None
