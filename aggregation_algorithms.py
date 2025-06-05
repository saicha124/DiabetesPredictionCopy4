import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import copy

class FedAvgAggregator:
    """Federated Averaging (FedAvg) aggregation algorithm"""
    
    def __init__(self):
        self.name = "FedAvg"
    
    def aggregate(self, global_model, client_updates):
        """Aggregate client updates using weighted averaging"""
        if not client_updates:
            return global_model
        
        try:
            # Calculate total samples
            total_samples = sum(update['num_samples'] for update in client_updates)
            
            if total_samples == 0:
                return global_model
            
            # Initialize aggregated parameters
            aggregated_params = None
            
            # Weighted aggregation
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                params = update['parameters']
                
                if aggregated_params is None:
                    aggregated_params = weight * params
                else:
                    aggregated_params += weight * params
            
            # Update global model with aggregated parameters
            updated_model = self._update_model_parameters(global_model, aggregated_params)
            
            return updated_model
            
        except Exception as e:
            print(f"FedAvg aggregation error: {e}")
            return global_model
    
    def _update_model_parameters(self, model, parameters):
        """Update model parameters"""
        try:
            updated_model = copy.deepcopy(model)
            
            if hasattr(updated_model, 'coef_') and hasattr(updated_model, 'intercept_'):
                # For logistic regression
                n_features = updated_model.coef_.shape[1]
                
                if len(parameters) >= n_features + 1:
                    updated_model.coef_ = parameters[:n_features].reshape(1, -1)
                    updated_model.intercept_ = parameters[n_features:n_features+1]
            
            return updated_model
            
        except Exception as e:
            print(f"Parameter update error: {e}")
            return model

class FedProxAggregator:
    """Federated Proximal (FedProx) aggregation algorithm"""
    
    def __init__(self, mu=0.01):
        self.name = "FedProx"
        self.mu = mu  # Proximal term coefficient
    
    def aggregate(self, global_model, client_updates):
        """Aggregate client updates with proximal regularization"""
        if not client_updates:
            return global_model
        
        try:
            # Get global model parameters for proximal term
            global_params = self._extract_model_parameters(global_model)
            
            # Calculate total samples
            total_samples = sum(update['num_samples'] for update in client_updates)
            
            if total_samples == 0:
                return global_model
            
            # Initialize aggregated parameters
            aggregated_params = None
            
            # Weighted aggregation with proximal regularization
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                params = update['parameters']
                
                # Apply proximal regularization
                if global_params is not None and len(params) == len(global_params):
                    regularized_params = params - self.mu * (params - global_params)
                else:
                    regularized_params = params
                
                if aggregated_params is None:
                    aggregated_params = weight * regularized_params
                else:
                    aggregated_params += weight * regularized_params
            
            # Update global model
            updated_model = self._update_model_parameters(global_model, aggregated_params)
            
            return updated_model
            
        except Exception as e:
            print(f"FedProx aggregation error: {e}")
            return global_model
    
    def _extract_model_parameters(self, model):
        """Extract parameters from model"""
        try:
            if hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
                return np.concatenate([
                    model.coef_.flatten(),
                    model.intercept_.flatten()
                ])
            return None
        except:
            return None
    
    def _update_model_parameters(self, model, parameters):
        """Update model parameters"""
        try:
            updated_model = copy.deepcopy(model)
            
            if hasattr(updated_model, 'coef_') and hasattr(updated_model, 'intercept_'):
                # For logistic regression
                n_features = updated_model.coef_.shape[1]
                
                if len(parameters) >= n_features + 1:
                    updated_model.coef_ = parameters[:n_features].reshape(1, -1)
                    updated_model.intercept_ = parameters[n_features:n_features+1]
            
            return updated_model
            
        except Exception as e:
            print(f"Parameter update error: {e}")
            return model

class SecureAggregator:
    """Secure aggregation with additional security measures"""
    
    def __init__(self, base_aggregator='FedAvg', threshold=0.1):
        self.threshold = threshold
        
        if base_aggregator == 'FedAvg':
            self.base_aggregator = FedAvgAggregator()
        else:
            self.base_aggregator = FedProxAggregator()
    
    def aggregate(self, global_model, client_updates):
        """Secure aggregation with anomaly detection"""
        if not client_updates:
            return global_model
        
        # Filter out anomalous updates
        filtered_updates = self._filter_anomalous_updates(client_updates)
        
        # Use base aggregator
        return self.base_aggregator.aggregate(global_model, filtered_updates)
    
    def _filter_anomalous_updates(self, client_updates):
        """Filter out potentially malicious updates"""
        if len(client_updates) <= 1:
            return client_updates
        
        try:
            # Calculate parameter statistics
            all_params = [update['parameters'] for update in client_updates]
            param_means = [np.mean(params) for params in all_params]
            
            overall_mean = np.mean(param_means)
            overall_std = np.std(param_means)
            
            # Filter updates that are too far from the mean
            filtered_updates = []
            for i, update in enumerate(client_updates):
                param_mean = param_means[i]
                
                if overall_std == 0 or abs(param_mean - overall_mean) <= 2 * overall_std:
                    filtered_updates.append(update)
            
            return filtered_updates if filtered_updates else client_updates
            
        except Exception as e:
            print(f"Anomaly filtering error: {e}")
            return client_updates
