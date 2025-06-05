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
        self.client_status = {}  # Track individual client training status
        
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
        if len(X) > 0:
            sample_size = min(100, len(X))
            sample_X, sample_y = X[:sample_size], y[:sample_size]
            self.global_model.fit(sample_X, sample_y)
    
    def _partition_data(self, X, y):
        """Partition data among clients ensuring balanced classes"""
        client_data = []
        n_samples = len(X)
        
        # Get class distribution
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_samples_per_client = max(10, n_samples // (self.num_clients * 2))
        
        # Create stratified partitions to ensure each client has both classes
        indices_by_class = {}
        for cls in unique_classes:
            indices_by_class[cls] = np.where(y == cls)[0]
            np.random.shuffle(indices_by_class[cls])
        
        # Distribute samples to clients ensuring class balance
        for i in range(self.num_clients):
            client_indices = []
            
            # Add samples from each class to this client
            for cls in unique_classes:
                class_indices = indices_by_class[cls]
                samples_per_client = len(class_indices) // self.num_clients
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                
                # For the last client, take remaining samples
                if i == self.num_clients - 1:
                    end_idx = len(class_indices)
                
                client_indices.extend(class_indices[start_idx:end_idx])
            
            # Ensure minimum samples per client
            if len(client_indices) < min_samples_per_client:
                # Add more samples if needed
                remaining_indices = []
                for cls in unique_classes:
                    remaining = set(indices_by_class[cls]) - set(client_indices)
                    remaining_indices.extend(list(remaining)[:5])  # Add up to 5 more per class
                client_indices.extend(remaining_indices[:min_samples_per_client - len(client_indices)])
            
            # Get data for this client
            client_indices = np.array(client_indices)
            np.random.shuffle(client_indices)
            
            client_X = X[client_indices]
            client_y = y[client_indices]
            
            # Split into train/test ensuring both sets have both classes if possible
            if len(client_X) >= 4 and len(np.unique(client_y)) > 1:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        client_X, client_y, test_size=0.3, random_state=42 + i, stratify=client_y
                    )
                except ValueError:
                    # Fallback to simple split if stratification fails
                    split_idx = len(client_X) * 7 // 10
                    X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                    y_train, y_test = client_y[:split_idx], client_y[split_idx:]
            else:
                # Simple split for small datasets
                split_idx = max(1, len(client_X) * 7 // 10)
                X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                y_train, y_test = client_y[:split_idx], client_y[split_idx:]
            
            # Ensure we have data for both train and test
            if len(X_train) == 0:
                X_train, y_train = client_X[:1], client_y[:1]
            if len(X_test) == 0:
                X_test, y_test = client_X[-1:], client_y[-1:]
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
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
                
                # Check for convergence and early stopping
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    
                # Update session state for best accuracy tracking
                with self.lock:
                    if 'best_accuracy' in st.session_state:
                        st.session_state.best_accuracy = self.best_accuracy
                
                if accuracy >= self.target_accuracy:
                    print(f"🎯 Target accuracy {self.target_accuracy:.3f} reached at round {self.current_round}!")
                    with self.lock:
                        if 'early_stopped' in st.session_state:
                            st.session_state.early_stopped = True
                    break
                
                # Simulate some delay for demonstration
                time.sleep(1)
            
            # Prepare final results with additional metrics
            total_time = sum([m['execution_time'] for m in self.training_history])
            target_reached = self.best_accuracy >= self.target_accuracy
            
            results = {
                'accuracy': self.best_accuracy,
                'final_accuracy': self.best_accuracy,
                'final_loss': self.training_history[-1]['loss'] if self.training_history else 0,
                'f1_score': self.training_history[-1]['f1_score'] if self.training_history else 0,
                'rounds_completed': self.current_round,
                'total_time': total_time,
                'training_history': self.training_history,
                'target_reached': target_reached,
                'early_stopped': target_reached,
                'best_accuracy': self.best_accuracy
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Training failed: {str(e)}")
    
    def _train_clients_parallel(self):
        """Train all clients in parallel"""
        client_updates = []
        
        def train_single_client(client):
            try:
                # Update client status
                with self.lock:
                    self.client_status[client.client_id] = 'training'
                    if 'client_status' in st.session_state:
                        st.session_state.client_status = self.client_status.copy()
                
                # Send global model parameters to client
                client.receive_global_model(self.global_model)
                
                # Train client model
                update = client.train()
                
                # Update client status
                with self.lock:
                    self.client_status[client.client_id] = 'completed'
                    if 'client_status' in st.session_state:
                        st.session_state.client_status = self.client_status.copy()
                
                return update
            except Exception as e:
                with self.lock:
                    self.client_status[client.client_id] = 'failed'
                    if 'client_status' in st.session_state:
                        st.session_state.client_status = self.client_status.copy()
                print(f"Client {client.client_id} training failed: {e}")
                return None
        
        # Initialize client status
        for client in self.clients:
            self.client_status[client.client_id] = 'waiting'
        
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
        if self.global_model is None:
            return 0.0, 1.0, 0.0, np.zeros((2, 2))
            
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for client in self.clients:
            X_test = client.data['X_test']
            y_test = client.data['y_test']
            
            if len(X_test) > 0 and self.global_model is not None:
                try:
                    predictions = self.global_model.predict(X_test)
                    probabilities = self.global_model.predict_proba(X_test)[:, 1]
                    
                    all_predictions.extend(predictions)
                    all_true_labels.extend(y_test)
                    all_probabilities.extend(probabilities)
                except Exception as e:
                    print(f"Error evaluating client {client.client_id}: {e}")
                    continue
        
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
