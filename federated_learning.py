import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import streamlit as st

from client_simulator import ClientSimulator
from aggregation_algorithms import FedAvgAggregator, FedProxAggregator
from differential_privacy import DifferentialPrivacyManager
from data_preprocessing import DataPreprocessor
from utils import calculate_metrics

class FederatedLearningManager:
    """Main federated learning orchestrator"""
    
    def __init__(self, num_clients=5, max_rounds=20, target_accuracy=0.85,
                 aggregation_algorithm='FedAvg', enable_dp=True, epsilon=1.0, 
                 delta=1e-5, committee_size=3):
        self.num_clients = num_clients
        self.max_rounds = max_rounds
        self.target_accuracy = target_accuracy
        self.aggregation_algorithm = aggregation_algorithm
        self.enable_dp = enable_dp
        self.epsilon = epsilon
        self.delta = delta
        self.committee_size = committee_size
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.dp_manager = DifferentialPrivacyManager(epsilon, delta) if enable_dp else None
        
        # Initialize aggregator
        if aggregation_algorithm == 'FedAvg':
            self.aggregator = FedAvgAggregator()
        else:
            self.aggregator = FedProxAggregator()
        
        # Training state
        self.current_round = 0
        self.global_model = None
        self.clients = []
        self.training_history = []
        self.best_accuracy = 0.0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def setup_clients(self, data):
        """Setup federated clients with data partitions"""
        # Preprocess data
        X, y = self.preprocessor.fit_transform(data)
        
        # Create data partitions for clients
        client_data = self._partition_data(X, y)
        
        # Initialize clients
        self.clients = []
        for i in range(self.num_clients):
            client = ClientSimulator(
                client_id=i,
                data=client_data[i],
                model_type='logistic_regression'
            )
            self.clients.append(client)
        
        # Initialize global model
        self.global_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear'
        )
        
        # Fit global model on a small sample to initialize parameters
        sample_X, sample_y = X[:100], y[:100]
        self.global_model.fit(sample_X, sample_y)
    
    def _partition_data(self, X, y):
        """Partition data among clients (non-IID distribution)"""
        client_data = []
        n_samples = len(X)
        
        # Create different partition sizes to simulate real-world scenario
        partition_sizes = np.random.dirichlet([1] * self.num_clients) * n_samples
        partition_sizes = partition_sizes.astype(int)
        
        # Ensure we use all data
        partition_sizes[-1] = n_samples - sum(partition_sizes[:-1])
        
        start_idx = 0
        for i, size in enumerate(partition_sizes):
            end_idx = start_idx + size
            if end_idx > n_samples:
                end_idx = n_samples
            
            client_X = X[start_idx:end_idx]
            client_y = y[start_idx:end_idx]
            
            # Split into train/test for each client
            if len(client_X) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    client_X, client_y, test_size=0.2, random_state=42, stratify=client_y
                )
            else:
                X_train, X_test, y_train, y_test = client_X, client_X, client_y, client_y
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
            
            start_idx = end_idx
            if start_idx >= n_samples:
                break
        
        return client_data
    
    def train(self, data):
        """Main federated training loop"""
        try:
            # Setup clients
            self.setup_clients(data)
            
            # Training loop
            for round_num in range(self.max_rounds):
                self.current_round = round_num + 1
                
                start_time = time.time()
                
                # Parallel client training
                client_updates = self._train_clients_parallel()
                
                # Committee-based security check
                validated_updates = self._committee_validation(client_updates)
                
                # Apply differential privacy
                if self.enable_dp and self.dp_manager:
                    validated_updates = self.dp_manager.add_noise(validated_updates)
                
                # Aggregate updates
                self.global_model = self.aggregator.aggregate(
                    self.global_model, validated_updates
                )
                
                # Evaluate global model
                accuracy, loss, f1, cm = self._evaluate_global_model()
                
                # Record metrics
                round_time = time.time() - start_time
                metrics = {
                    'round': self.current_round,
                    'accuracy': accuracy,
                    'loss': loss,
                    'f1_score': f1,
                    'execution_time': round_time
                }
                
                # Update session state for real-time monitoring
                with self.lock:
                    if 'training_metrics' in st.session_state:
                        st.session_state.training_metrics.append(metrics)
                    if 'confusion_matrices' in st.session_state:
                        st.session_state.confusion_matrices.append(cm)
                    if 'execution_times' in st.session_state:
                        st.session_state.execution_times.append(round_time)
                    if 'communication_times' in st.session_state:
                        # Simulate communication time
                        comm_time = np.random.normal(0.5, 0.1)
                        st.session_state.communication_times.append(max(0.1, comm_time))
                
                self.training_history.append(metrics)
                
                # Check for convergence
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                
                if accuracy >= self.target_accuracy:
                    break
                
                # Simulate some delay for demonstration
                time.sleep(1)
            
            # Prepare final results
            total_time = sum([m['execution_time'] for m in self.training_history])
            results = {
                'final_accuracy': self.best_accuracy,
                'final_loss': self.training_history[-1]['loss'] if self.training_history else 0,
                'rounds_completed': self.current_round,
                'total_time': total_time,
                'training_history': self.training_history
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def _train_clients_parallel(self):
        """Train all clients in parallel"""
        client_updates = []
        
        def train_single_client(client):
            try:
                # Send global model parameters to client
                client.receive_global_model(self.global_model)
                
                # Train client model
                update = client.train()
                
                return update
            except Exception as e:
                print(f"Client {client.client_id} training failed: {e}")
                return None
        
        # Execute parallel training
        with ThreadPoolExecutor(max_workers=min(self.num_clients, 4)) as executor:
            futures = [executor.submit(train_single_client, client) for client in self.clients]
            
            for future in as_completed(futures):
                try:
                    update = future.result()
                    if update is not None:
                        client_updates.append(update)
                except Exception as e:
                    print(f"Client training error: {e}")
        
        return client_updates
    
    def _committee_validation(self, client_updates):
        """Committee-based security validation"""
        if len(client_updates) < self.committee_size:
            return client_updates
        
        # Select committee members randomly
        committee_indices = np.random.choice(
            len(client_updates), 
            size=min(self.committee_size, len(client_updates)), 
            replace=False
        )
        
        validated_updates = []
        
        for i, update in enumerate(client_updates):
            if i in committee_indices:
                # Committee members automatically validated
                validated_updates.append(update)
            else:
                # Validate against committee consensus
                if self._validate_update(update, [client_updates[j] for j in committee_indices]):
                    validated_updates.append(update)
        
        return validated_updates
    
    def _validate_update(self, update, committee_updates):
        """Validate an update against committee consensus"""
        if not committee_updates:
            return True
        
        # Simple validation: check if parameters are within reasonable bounds
        try:
            update_params = update['parameters']
            committee_params = [u['parameters'] for u in committee_updates]
            
            # Calculate mean and std of committee parameters
            committee_mean = np.mean([np.mean(params) for params in committee_params])
            committee_std = np.std([np.mean(params) for params in committee_params])
            
            update_mean = np.mean(update_params)
            
            # Check if update is within 2 standard deviations
            threshold = 2 * committee_std if committee_std > 0 else 1.0
            
            return abs(update_mean - committee_mean) <= threshold
            
        except Exception:
            # If validation fails, reject the update
            return False
    
    def _evaluate_global_model(self):
        """Evaluate global model on all clients' test data"""
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for client in self.clients:
            X_test = client.data['X_test']
            y_test = client.data['y_test']
            
            if len(X_test) > 0:
                predictions = self.global_model.predict(X_test)
                probabilities = self.global_model.predict_proba(X_test)[:, 1]
                
                all_predictions.extend(predictions)
                all_true_labels.extend(y_test)
                all_probabilities.extend(probabilities)
        
        if len(all_predictions) == 0:
            return 0.0, 1.0, 0.0, np.zeros((2, 2))
        
        # Calculate metrics
        accuracy = accuracy_score(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions, average='weighted')
        
        # Calculate loss
        try:
            loss = log_loss(all_true_labels, all_probabilities)
        except:
            loss = 1.0
        
        # Confusion matrix
        cm = confusion_matrix(all_true_labels, all_predictions)
        
        return accuracy, loss, f1, cm
