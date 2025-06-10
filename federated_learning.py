import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss
import time
import math
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
                 delta=1e-5, committee_size=3, model_type='logistic_regression',
                 privacy_mechanism='gaussian', gradient_clip_norm=1.0):
        self.num_clients = num_clients
        self.max_rounds = max_rounds
        self.target_accuracy = target_accuracy
        self.aggregation_algorithm = aggregation_algorithm
        self.enable_dp = enable_dp
        self.epsilon = epsilon
        self.delta = delta
        self.committee_size = committee_size
        self.model_type = model_type
        self.privacy_mechanism = privacy_mechanism
        self.gradient_clip_norm = gradient_clip_norm
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        if enable_dp:
            self.dp_manager = DifferentialPrivacyManager(
                epsilon=epsilon, 
                delta=delta, 
                mechanism=privacy_mechanism,
                sensitivity=1.0
            )
            if hasattr(self.dp_manager, 'gradient_clip_norm'):
                self.dp_manager.gradient_clip_norm = gradient_clip_norm
        else:
            self.dp_manager = None
        
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
        
        # Initialize convergence tracking attributes
        self.convergence_reason = None  # Track why training stopped ('model_convergence' or 'max_rounds_reached')
        self.early_stopped = False
        
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
    
    def setup_clients_with_data(self, client_data):
        """Setup federated clients with pre-distributed data"""
        from client_simulator import ClientSimulator
        from sklearn.linear_model import LogisticRegression
        
        # Create client instances with provided data
        self.clients = []
        for i, data_partition in enumerate(client_data):
            client = ClientSimulator(
                client_id=i,
                data=data_partition,
                model_type='logistic_regression'
            )
            self.clients.append(client)
        
        # Initialize global model
        self.global_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            solver='liblinear'
        )
        
        # Initialize global model with sample data from first non-empty client
        for data_partition in client_data:
            if len(data_partition['X_train']) > 0:
                sample_X = data_partition['X_train'][:min(50, len(data_partition['X_train']))]
                sample_y = data_partition['y_train'][:min(50, len(data_partition['y_train']))]
                if len(sample_X) > 0 and len(np.unique(sample_y)) > 1:
                    try:
                        self.global_model.fit(sample_X, sample_y)
                        break
                    except Exception as e:
                        print(f"Failed to initialize global model: {e}")
                        continue
    
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
                
                # Apply differential privacy with current parameters
                if self.enable_dp and self.dp_manager:
                    # Update privacy parameters if they changed in session state
                    if hasattr(st, 'session_state'):
                        current_epsilon = st.session_state.get('epsilon', self.epsilon)
                        current_delta = st.session_state.get('delta', self.delta)
                        
                        # Update DP manager parameters if they changed
                        if current_epsilon != self.dp_manager.epsilon or current_delta != self.dp_manager.delta:
                            self.dp_manager.epsilon = current_epsilon
                            self.dp_manager.delta = current_delta
                            self.dp_manager.noise_scale = self.dp_manager._calculate_noise_scale()
                            print(f"Updated privacy parameters: ε={current_epsilon}, δ={current_delta}, noise_scale={self.dp_manager.noise_scale}")
                    
                    # Add noise to validated updates with adaptive sensitivity
                    print(f"Applying DP noise with ε={self.dp_manager.epsilon}, noise_scale={self.dp_manager.noise_scale}")
                    noisy_updates = []
                    for update in validated_updates:
                        if 'parameters' in update and isinstance(update['parameters'], np.ndarray):
                            original_params = update['parameters'].copy()
                            
                            # Calculate adaptive sensitivity based on parameter magnitude
                            param_magnitude = np.linalg.norm(original_params)
                            adaptive_sensitivity = max(0.1, param_magnitude * 0.1)  # 10% of parameter magnitude
                            
                            # Calculate noise scale with adaptive sensitivity
                            if self.dp_manager.delta == 0:
                                noise_scale = adaptive_sensitivity / self.dp_manager.epsilon
                            else:
                                import math
                                c = math.sqrt(2 * math.log(1.25 / self.dp_manager.delta))
                                noise_scale = c * adaptive_sensitivity / self.dp_manager.epsilon
                            
                            # Add Gaussian noise
                            noise = np.random.normal(0, noise_scale, size=original_params.shape)
                            noisy_params = original_params + noise
                            
                            noisy_update = update.copy()
                            noisy_update['parameters'] = noisy_params
                            noisy_update['dp_applied'] = True
                            noisy_update['epsilon'] = self.dp_manager.epsilon
                            noisy_update['noise_magnitude'] = np.linalg.norm(noise)
                            noisy_update['adaptive_sensitivity'] = adaptive_sensitivity
                            noisy_update['noise_scale_used'] = noise_scale
                            
                            print(f"Client {update.get('client_id', 'unknown')}: ε={self.dp_manager.epsilon}, sensitivity={adaptive_sensitivity:.4f}, noise_scale={noise_scale:.4f}, noise_mag={np.linalg.norm(noise):.6f}")
                            noisy_updates.append(noisy_update)
                        else:
                            noisy_updates.append(update)
                    
                    validated_updates = noisy_updates
                
                # Aggregate updates
                self.global_model = self.aggregator.aggregate(
                    self.global_model, validated_updates
                )
                
                # Evaluate global model
                accuracy, loss, f1, cm = self._evaluate_global_model()
                
                # Record metrics
                round_time = time.time() - start_time
                
                # Calculate DP effects if applied
                dp_effects = {}
                if self.enable_dp and validated_updates:
                    dp_applied_count = sum(1 for update in validated_updates if update.get('dp_applied', False))
                    avg_noise_magnitude = np.mean([update.get('noise_magnitude', 0) for update in validated_updates if 'noise_magnitude' in update]) if validated_updates else 0
                    dp_effects = {
                        'dp_noise_applied': dp_applied_count,
                        'avg_noise_magnitude': avg_noise_magnitude,
                        'epsilon_used': self.dp_manager.epsilon if self.dp_manager else 0
                    }
                
                metrics = {
                    'round': self.current_round,
                    'accuracy': accuracy,
                    'loss': loss,
                    'f1_score': f1,
                    'execution_time': round_time,
                    **dp_effects
                }
                
                # Store metrics in training history (no session state access)
                self.training_history.append(metrics)
                
                # Store confusion matrix
                if not hasattr(self, 'confusion_matrices'):
                    self.confusion_matrices = []
                self.confusion_matrices.append(cm)
                
                # Store execution times
                if not hasattr(self, 'execution_times'):
                    self.execution_times = []
                self.execution_times.append(round_time)
                
                # Store communication times
                if not hasattr(self, 'communication_times'):
                    self.communication_times = []
                comm_time = np.random.normal(0.5, 0.1)
                self.communication_times.append(max(0.1, comm_time))
                
                # ============================================================================
                # GLOBAL STOPPING CRITERIA - Two main conditions for stopping training:
                # 1. Model convergence (when performance plateaus over consecutive rounds)
                # 2. Maximum rounds reached (computational budget exhausted)
                # ============================================================================
                
                # Update best accuracy achieved so far
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                
                # STOPPING CRITERION 1: CONVERGENCE DETECTION
                # Check if the global model has converged by analyzing performance trends
                convergence_detected = self._check_global_convergence()
                
                if convergence_detected:
                    print(f"🔄 Global model converged at round {self.current_round}! "
                          f"Best accuracy: {self.best_accuracy:.3f}")
                    self.early_stopped = True
                    self.convergence_reason = "model_convergence"
                    break
                
                # STOPPING CRITERION 2: MAXIMUM ROUNDS CHECK
                # This will be handled by the main loop condition, but we track it here
                if self.current_round >= self.max_rounds:
                    print(f"📊 Maximum rounds ({self.max_rounds}) reached! "
                          f"Final accuracy: {accuracy:.3f}")
                    self.convergence_reason = "max_rounds_reached"
                    # Loop will naturally break due to range condition
                
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
    
    def _check_global_convergence(self):
        """
        Check if the global model has converged based on performance trends.
        
        CONVERGENCE DETECTION ALGORITHM:
        ================================
        The global model is considered converged when:
        1. We have sufficient training history (minimum 3 rounds)
        2. Performance improvement has plateaued over consecutive rounds
        3. Both accuracy and loss show minimal improvement
        
        CONVERGENCE CRITERIA:
        ====================
        - Accuracy improvement < 0.5% for last 3 consecutive rounds
        - Loss improvement < 0.01 for last 3 consecutive rounds
        - No significant oscillation in performance metrics
        
        Returns:
            bool: True if convergence detected, False otherwise
        """
        
        # STEP 1: Check if we have sufficient training history
        # Need at least 3 rounds to detect convergence trends
        min_rounds_for_convergence = 3
        if len(self.training_history) < min_rounds_for_convergence:
            return False
        
        # STEP 2: Extract recent performance metrics
        # Get the last 3 rounds of training metrics for trend analysis
        recent_rounds = self.training_history[-3:]
        recent_accuracies = [round_data['accuracy'] for round_data in recent_rounds]
        recent_losses = [round_data['loss'] for round_data in recent_rounds]
        
        # STEP 3: Calculate accuracy improvement trends
        # Check if accuracy improvements are below convergence threshold
        accuracy_improvements = []
        for i in range(1, len(recent_accuracies)):
            improvement = recent_accuracies[i] - recent_accuracies[i-1]
            accuracy_improvements.append(improvement)
        
        # STEP 4: Calculate loss improvement trends  
        # Check if loss reductions are below convergence threshold
        loss_improvements = []
        for i in range(1, len(recent_losses)):
            improvement = recent_losses[i-1] - recent_losses[i]  # Loss should decrease
            loss_improvements.append(improvement)
        
        # STEP 5: Define convergence thresholds
        # These thresholds determine when improvements are considered negligible
        accuracy_convergence_threshold = 0.005  # 0.5% accuracy improvement
        loss_convergence_threshold = 0.01       # 0.01 loss improvement
        
        # STEP 6: Check accuracy convergence condition
        # All recent accuracy improvements must be below threshold
        accuracy_converged = all(
            abs(improvement) < accuracy_convergence_threshold 
            for improvement in accuracy_improvements
        )
        
        # STEP 7: Check loss convergence condition
        # All recent loss improvements must be below threshold
        loss_converged = all(
            improvement < loss_convergence_threshold 
            for improvement in loss_improvements
        )
        
        # STEP 8: Check for performance oscillation
        # Detect if metrics are oscillating rather than converging
        accuracy_variance = np.var(recent_accuracies)
        oscillation_threshold = 0.001  # Low variance indicates stability
        stable_performance = accuracy_variance < oscillation_threshold
        
        # STEP 9: Final convergence decision
        # Model is converged if both accuracy and loss have plateaued with stable performance
        convergence_detected = accuracy_converged and loss_converged and stable_performance
        
        # STEP 10: Log convergence analysis for debugging
        if convergence_detected:
            print(f"🔍 CONVERGENCE DETECTED:")
            print(f"   - Recent accuracies: {[f'{acc:.4f}' for acc in recent_accuracies]}")
            print(f"   - Recent losses: {[f'{loss:.4f}' for loss in recent_losses]}")
            print(f"   - Accuracy improvements: {[f'{imp:.4f}' for imp in accuracy_improvements]}")
            print(f"   - Loss improvements: {[f'{imp:.4f}' for imp in loss_improvements]}")
            print(f"   - Performance variance: {accuracy_variance:.6f}")
        
        return convergence_detected
