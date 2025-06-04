import numpy as np
import random
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import hashlib
import time

class SecurityCommittee:
    """Committee-based security system for federated learning"""
    
    def __init__(self, privacy_engine=None, reputation_threshold: float = 0.5):
        """
        Initialize security committee system
        
        Args:
            privacy_engine: Privacy engine for DP operations
            reputation_threshold: Minimum reputation for committee membership
        """
        self.privacy_engine = privacy_engine
        self.reputation_threshold = reputation_threshold
        self.committee_history = []
        self.client_reputation_history = defaultdict(list)
        self.suspicious_clients = set()
        self.blacklisted_clients = set()
        
        # Committee parameters
        self.min_committee_size = 3
        self.max_committee_size = 10
        self.rotation_frequency = 5  # Rotate committee every N rounds
        
        print("Security committee system initialized")
    
    def select_committee(self, reputation_scores: List[Tuple[int, float]], 
                        committee_size: int = 5) -> List[int]:
        """
        Select committee members based on DP-masked reputation scores
        
        Args:
            reputation_scores: List of (client_id, noisy_reputation) tuples
            committee_size: Desired committee size
            
        Returns:
            List of selected committee member IDs
        """
        try:
            # Filter out blacklisted clients
            eligible_clients = [
                (client_id, score) for client_id, score in reputation_scores
                if client_id not in self.blacklisted_clients and score >= self.reputation_threshold
            ]
            
            if len(eligible_clients) < self.min_committee_size:
                # If not enough eligible clients, relax threshold
                eligible_clients = [
                    (client_id, score) for client_id, score in reputation_scores
                    if client_id not in self.blacklisted_clients
                ]
            
            if not eligible_clients:
                return []
            
            # Sort by reputation score (descending)
            eligible_clients.sort(key=lambda x: x[1], reverse=True)
            
            # Select top candidates with some randomization for security
            committee_size = min(committee_size, len(eligible_clients))
            
            # Select top 70% deterministically, remaining 30% randomly
            deterministic_count = max(1, int(0.7 * committee_size))
            random_count = committee_size - deterministic_count
            
            committee_members = []
            
            # Deterministic selection (top performers)
            for i in range(min(deterministic_count, len(eligible_clients))):
                committee_members.append(eligible_clients[i][0])
            
            # Random selection from remaining candidates
            if random_count > 0 and len(eligible_clients) > deterministic_count:
                remaining_candidates = eligible_clients[deterministic_count:]
                random_selections = random.sample(
                    remaining_candidates, 
                    min(random_count, len(remaining_candidates))
                )
                committee_members.extend([client_id for client_id, _ in random_selections])
            
            # Record committee formation
            self.committee_history.append({
                'members': committee_members,
                'timestamp': time.time(),
                'selection_method': 'reputation_based'
            })
            
            return committee_members
            
        except Exception as e:
            print(f"Committee selection failed: {e}")
            return []
    
    def validate_client_update(self, client_id: int, update: Dict) -> Dict:
        """
        Validate a client's update for potential attacks
        
        Args:
            client_id: ID of the client
            update: Client's model update
            
        Returns:
            Validation result dictionary
        """
        try:
            validation_result = {
                'client_id': client_id,
                'is_valid': True,
                'suspicious_score': 0.0,
                'attack_indicators': [],
                'recommendation': 'accept'
            }
            
            # Check for Byzantine behavior
            byzantine_score = self._detect_byzantine_behavior(client_id, update)
            validation_result['suspicious_score'] += byzantine_score
            
            if byzantine_score > 0.5:
                validation_result['attack_indicators'].append('byzantine_behavior')
            
            # Check for Sybil attack patterns
            sybil_score = self._detect_sybil_patterns(client_id, update)
            validation_result['suspicious_score'] += sybil_score
            
            if sybil_score > 0.3:
                validation_result['attack_indicators'].append('sybil_patterns')
            
            # Check update quality
            quality_score = self._assess_update_quality(update)
            if quality_score < 0.3:
                validation_result['suspicious_score'] += 0.2
                validation_result['attack_indicators'].append('poor_quality')
            
            # Make final decision
            if validation_result['suspicious_score'] > 0.7:
                validation_result['is_valid'] = False
                validation_result['recommendation'] = 'reject'
                self.suspicious_clients.add(client_id)
            elif validation_result['suspicious_score'] > 0.4:
                validation_result['recommendation'] = 'investigate'
                self.suspicious_clients.add(client_id)
            
            return validation_result
            
        except Exception as e:
            print(f"Update validation failed for client {client_id}: {e}")
            return {
                'client_id': client_id,
                'is_valid': True,
                'suspicious_score': 0.0,
                'attack_indicators': [],
                'recommendation': 'accept'
            }
    
    def _detect_byzantine_behavior(self, client_id: int, update: Dict) -> float:
        """
        Detect Byzantine attack patterns
        
        Args:
            client_id: Client ID
            update: Client update
            
        Returns:
            Suspicion score (0-1)
        """
        try:
            suspicion_score = 0.0
            
            # Check for extreme parameter values
            if 'weights' in update:
                weights = update['weights']
                for param_name, param_tensor in weights.items():
                    if hasattr(param_tensor, 'numpy'):
                        param_array = param_tensor.detach().numpy()
                    else:
                        param_array = np.array(param_tensor)
                    
                    # Check for extreme values
                    if np.any(np.abs(param_array) > 100):
                        suspicion_score += 0.3
                    
                    # Check for NaN or inf values
                    if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                        suspicion_score += 0.5
            
            # Check loss values
            if 'loss' in update:
                loss = update['loss']
                if loss < 0 or loss > 100 or np.isnan(loss) or np.isinf(loss):
                    suspicion_score += 0.4
            
            # Check accuracy values
            if 'accuracy' in update:
                accuracy = update['accuracy']
                if accuracy < 0 or accuracy > 1 or np.isnan(accuracy):
                    suspicion_score += 0.3
            
            return min(1.0, suspicion_score)
            
        except Exception as e:
            print(f"Byzantine detection failed: {e}")
            return 0.0
    
    def _detect_sybil_patterns(self, client_id: int, update: Dict) -> float:
        """
        Detect Sybil attack patterns
        
        Args:
            client_id: Client ID
            update: Client update
            
        Returns:
            Suspicion score (0-1)
        """
        try:
            suspicion_score = 0.0
            
            # Check for identical or very similar updates
            if len(self.committee_history) > 0:
                # Compare with recent updates (simplified similarity check)
                if 'loss' in update and 'accuracy' in update:
                    current_loss = update['loss']
                    current_accuracy = update['accuracy']
                    
                    # Check client's history for suspicious consistency
                    client_history = self.client_reputation_history.get(client_id, [])
                    if len(client_history) >= 3:
                        recent_losses = [h.get('loss', 0) for h in client_history[-3:]]
                        recent_accuracies = [h.get('accuracy', 0) for h in client_history[-3:]]
                        
                        # Check for suspicious consistency
                        loss_variance = np.var(recent_losses + [current_loss])
                        accuracy_variance = np.var(recent_accuracies + [current_accuracy])
                        
                        if loss_variance < 1e-6 and accuracy_variance < 1e-6:
                            suspicion_score += 0.4
            
            # Check update timing patterns
            current_time = time.time()
            if len(self.committee_history) > 0:
                last_update_time = self.committee_history[-1].get('timestamp', 0)
                time_diff = current_time - last_update_time
                
                # Suspicious if updates are too regular or too fast
                if time_diff < 0.1:  # Too fast
                    suspicion_score += 0.2
            
            return min(1.0, suspicion_score)
            
        except Exception as e:
            print(f"Sybil detection failed: {e}")
            return 0.0
    
    def _assess_update_quality(self, update: Dict) -> float:
        """
        Assess the quality of a client update
        
        Args:
            update: Client update
            
        Returns:
            Quality score (0-1)
        """
        try:
            quality_score = 1.0
            
            # Check if required fields are present
            required_fields = ['weights', 'loss', 'accuracy', 'num_samples']
            missing_fields = [field for field in required_fields if field not in update]
            
            if missing_fields:
                quality_score -= 0.2 * len(missing_fields)
            
            # Check reasonable loss values
            if 'loss' in update:
                loss = update['loss']
                if 0.01 <= loss <= 10:  # Reasonable range
                    quality_score += 0.1
                else:
                    quality_score -= 0.2
            
            # Check reasonable accuracy values
            if 'accuracy' in update:
                accuracy = update['accuracy']
                if 0.1 <= accuracy <= 1.0:  # Reasonable range
                    quality_score += 0.1
                else:
                    quality_score -= 0.2
            
            # Check number of samples
            if 'num_samples' in update:
                num_samples = update['num_samples']
                if num_samples > 0:
                    quality_score += 0.1
                else:
                    quality_score -= 0.3
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            print(f"Quality assessment failed: {e}")
            return 0.5
    
    def update_client_reputation(self, client_id: int, round_result: Dict):
        """
        Update client reputation based on round performance
        
        Args:
            client_id: Client ID
            round_result: Results from the training round
        """
        try:
            # Record client performance
            self.client_reputation_history[client_id].append({
                'loss': round_result.get('loss', 0),
                'accuracy': round_result.get('accuracy', 0),
                'timestamp': time.time(),
                'validation_result': round_result.get('validation_result', {})
            })
            
            # Maintain history limit
            if len(self.client_reputation_history[client_id]) > 10:
                self.client_reputation_history[client_id].pop(0)
            
        except Exception as e:
            print(f"Reputation update failed for client {client_id}: {e}")
    
    def get_security_stats(self) -> Dict:
        """
        Get current security statistics
        
        Returns:
            Dictionary of security statistics
        """
        return {
            'total_committees_formed': len(self.committee_history),
            'suspicious_clients': len(self.suspicious_clients),
            'blacklisted_clients': len(self.blacklisted_clients),
            'current_reputation_threshold': self.reputation_threshold,
            'committee_rotation_frequency': self.rotation_frequency
        }
    
    def blacklist_client(self, client_id: int, reason: str = "security_violation"):
        """
        Blacklist a client for security violations
        
        Args:
            client_id: Client to blacklist
            reason: Reason for blacklisting
        """
        self.blacklisted_clients.add(client_id)
        self.suspicious_clients.discard(client_id)
        
        print(f"Client {client_id} blacklisted: {reason}")
    
    def is_client_trusted(self, client_id: int) -> bool:
        """
        Check if a client is trusted
        
        Args:
            client_id: Client ID to check
            
        Returns:
            True if client is trusted
        """
        return (client_id not in self.blacklisted_clients and 
                client_id not in self.suspicious_clients)
    
    def get_committee_diversity_score(self, committee_members: List[int]) -> float:
        """
        Calculate diversity score of committee members
        
        Args:
            committee_members: List of committee member IDs
            
        Returns:
            Diversity score (0-1)
        """
        try:
            if len(committee_members) <= 1:
                return 0.0
            
            # Calculate diversity based on reputation history variance
            reputation_variances = []
            
            for member_id in committee_members:
                history = self.client_reputation_history.get(member_id, [])
                if len(history) >= 2:
                    losses = [h.get('loss', 0) for h in history]
                    variance = np.var(losses) if losses else 0
                    reputation_variances.append(variance)
            
            if not reputation_variances:
                return 0.5  # Default diversity score
            
            # Higher variance across members indicates more diversity
            overall_variance = np.var(reputation_variances)
            diversity_score = min(1.0, overall_variance * 10)  # Scale appropriately
            
            return diversity_score
            
        except Exception as e:
            print(f"Diversity score calculation failed: {e}")
            return 0.5
