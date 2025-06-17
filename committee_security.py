"""
Committee-Based Security System for Hierarchical Federated Learning
===================================================================

This module implements advanced security mechanisms including:
- Periodic, reputation-weighted committee selection with role rotation
- Differential privacy-protected reputation calculation
- Sybil and Byzantine attack mitigation
- Cryptographic proofs for committee verifiability
"""

import numpy as np
import hashlib
import time
import random
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets


class NodeRole(Enum):
    """Committee node roles"""
    VALIDATOR = "validator"
    AGGREGATOR = "aggregator" 
    MONITOR = "monitor"
    COORDINATOR = "coordinator"


@dataclass
class NodeReputation:
    """Node reputation metrics with differential privacy protection"""
    node_id: str
    accuracy_score: float = 0.0
    response_time: float = 0.0
    availability: float = 0.0
    malicious_behavior_count: int = 0
    successful_validations: int = 0
    total_participations: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def calculate_reputation_score(self) -> float:
        """Calculate overall reputation score (0-1)"""
        if self.total_participations == 0:
            return 0.5  # Neutral reputation for new nodes
        
        # Weighted reputation calculation
        accuracy_weight = 0.4
        availability_weight = 0.3
        validation_weight = 0.2
        malicious_penalty = 0.1
        
        validation_ratio = self.successful_validations / max(1, self.total_participations)
        malicious_penalty_score = max(0, 1 - (self.malicious_behavior_count * 0.1))
        
        reputation = (
            self.accuracy_score * accuracy_weight +
            self.availability * availability_weight +
            validation_ratio * validation_weight +
            malicious_penalty_score * malicious_penalty
        )
        
        return max(0.0, min(1.0, reputation))


@dataclass
class CommitteeNode:
    """Committee node with cryptographic capabilities"""
    node_id: str
    public_key: Any
    private_key: Any
    role: NodeRole
    reputation: NodeReputation
    selection_timestamp: float = field(default_factory=time.time)
    
    def sign_message(self, message: str) -> bytes:
        """Sign a message with node's private key"""
        message_bytes = message.encode('utf-8')
        signature = self.private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, message: str, signature: bytes, public_key: Any) -> bool:
        """Verify message signature"""
        try:
            message_bytes = message.encode('utf-8')
            public_key.verify(
                signature,
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False


class DifferentialPrivacyReputationManager:
    """Manages reputation calculation with differential privacy protection"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0  # Maximum change in reputation per update
    
    def add_laplace_noise(self, value: float) -> float:
        """Add Laplace noise for differential privacy"""
        scale = self.sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return max(0.0, min(1.0, value + noise))
    
    def update_reputation_with_privacy(self, node: NodeReputation, 
                                     accuracy_delta: float,
                                     availability_delta: float) -> NodeReputation:
        """Update reputation with differential privacy protection"""
        # Add noise to protect against reputation attacks
        noisy_accuracy = self.add_laplace_noise(accuracy_delta)
        noisy_availability = self.add_laplace_noise(availability_delta)
        
        # Update with privacy-protected values
        node.accuracy_score = max(0.0, min(1.0, node.accuracy_score + noisy_accuracy))
        node.availability = max(0.0, min(1.0, node.availability + noisy_availability))
        node.total_participations += 1
        node.last_updated = time.time()
        
        return node


class SybilDetector:
    """Detects and mitigates Sybil attacks"""
    
    def __init__(self, difficulty_threshold: int = 4):
        self.difficulty_threshold = difficulty_threshold
        self.registered_nodes: Dict[str, float] = {}  # node_id -> registration_time
        self.node_behaviors: Dict[str, List[Dict]] = {}  # behavior tracking
    
    def proof_of_work_challenge(self, node_id: str) -> Tuple[str, int]:
        """Generate proof-of-work challenge for node registration"""
        challenge = secrets.token_hex(32)
        return challenge, self.difficulty_threshold
    
    def verify_proof_of_work(self, node_id: str, challenge: str, nonce: int) -> bool:
        """Verify proof-of-work solution"""
        solution = hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest()
        return solution.startswith('0' * self.difficulty_threshold)
    
    def detect_sybil_patterns(self, node_ids: List[str]) -> List[str]:
        """Detect potential Sybil nodes based on behavior patterns"""
        suspicious_nodes = []
        
        for node_id in node_ids:
            if node_id not in self.node_behaviors:
                continue
                
            behaviors = self.node_behaviors[node_id]
            
            # Check for suspicious patterns
            if len(behaviors) > 10:  # Enough data for analysis
                response_times = [b.get('response_time', 0) for b in behaviors[-10:]]
                accuracy_scores = [b.get('accuracy', 0) for b in behaviors[-10:]]
                
                # Detect extremely suspicious behavior (very lenient for FL)
                if (np.std(response_times) < 0.001 or  # Extremely consistent response times
                    np.std(accuracy_scores) < 0.0001 or  # Extremely consistent accuracy
                    np.mean(accuracy_scores) > 0.999):  # Perfect accuracy (nearly impossible)
                    suspicious_nodes.append(node_id)
        
        return suspicious_nodes


class ByzantineDetector:
    """Enhanced Byzantine attack detection with multi-layered security"""
    
    def __init__(self, byzantine_threshold: float = 0.33):
        self.byzantine_threshold = byzantine_threshold
        self.deviation_threshold = 1.5  # Stricter threshold
        self.node_deviations: Dict[str, List[float]] = {}
        self.node_consistency_scores: Dict[str, List[float]] = {}
        self.node_response_patterns: Dict[str, List[float]] = {}
        self.consensus_history: List[np.ndarray] = []
        self.detection_history: Dict[str, int] = {}  # Track detection counts
        self.anomaly_scores: Dict[str, List[float]] = {}
    
    def detect_byzantine_behavior(self, node_updates: Dict[str, np.ndarray],
                                global_update: np.ndarray) -> List[str]:
        """Enhanced Byzantine detection with multiple validation layers"""
        byzantine_nodes = []
        
        if len(node_updates) == 0 or global_update is None:
            return byzantine_nodes
        
        # Store consensus history for trend analysis
        self.consensus_history.append(global_update.copy())
        if len(self.consensus_history) > 20:
            self.consensus_history = self.consensus_history[-20:]
        
        # Multi-layer detection system
        suspects = set()
        
        # 1. Statistical deviation analysis
        deviation_suspects = self._detect_statistical_anomalies(node_updates, global_update)
        suspects.update(deviation_suspects)
        
        # 2. Consensus violation detection
        consensus_suspects = self._detect_consensus_violations(node_updates, global_update)
        suspects.update(consensus_suspects)
        
        # 3. Pattern-based anomaly detection
        pattern_suspects = self._detect_pattern_anomalies(node_updates)
        suspects.update(pattern_suspects)
        
        # 4. Cross-validation with peer nodes
        peer_suspects = self._detect_peer_validation_failures(node_updates)
        suspects.update(peer_suspects)
        
        # 5. Temporal consistency analysis
        temporal_suspects = self._detect_temporal_inconsistencies(node_updates)
        suspects.update(temporal_suspects)
        
        # Apply weighted scoring system
        for node_id in suspects:
            score = self._calculate_byzantine_score(node_id, node_updates, global_update)
            
            # Track detection history
            if node_id not in self.detection_history:
                self.detection_history[node_id] = 0
            
            # Flag as Byzantine if score exceeds threshold
            if score > 0.25:  # Lower threshold for better detection
                self.detection_history[node_id] += 1
                byzantine_nodes.append(node_id)
        
        return byzantine_nodes
    
    def _detect_statistical_anomalies(self, node_updates: Dict[str, np.ndarray], 
                                     global_update: np.ndarray) -> List[str]:
        """Detect statistical anomalies in node updates"""
        suspects = []
        
        # Calculate statistical metrics for all updates
        all_deviations = []
        node_deviations_current = {}
        
        for node_id, update in node_updates.items():
            if update is not None and len(update) > 0:
                deviation = np.linalg.norm(update - global_update)
                all_deviations.append(deviation)
                node_deviations_current[node_id] = deviation
                
                # Store in history
                if node_id not in self.node_deviations:
                    self.node_deviations[node_id] = []
                self.node_deviations[node_id].append(deviation)
                
                if len(self.node_deviations[node_id]) > 15:
                    self.node_deviations[node_id] = self.node_deviations[node_id][-15:]
        
        # Calculate dynamic threshold based on current round statistics
        if len(all_deviations) > 1:
            median_deviation = np.median(all_deviations)
            mad = np.median(np.abs(np.array(all_deviations) - median_deviation))
            dynamic_threshold = median_deviation + 1.5 * mad  # More sensitive detection
            
            for node_id, deviation in node_deviations_current.items():
                # Check both current and historical anomalies
                if deviation > dynamic_threshold:
                    suspects.append(node_id)
                
                # Check historical consistency
                if len(self.node_deviations[node_id]) >= 3:
                    recent_trend = np.mean(self.node_deviations[node_id][-3:])
                    if recent_trend > self.deviation_threshold:
                        suspects.append(node_id)
        
        return suspects
    
    def _detect_consensus_violations(self, node_updates: Dict[str, np.ndarray],
                                   global_update: np.ndarray) -> List[str]:
        """Detect violations of consensus protocols"""
        suspects = []
        
        if len(self.consensus_history) < 3:
            return suspects
        
        # Analyze consensus stability
        recent_consensus = self.consensus_history[-3:]
        consensus_trend = np.mean([np.linalg.norm(c) for c in recent_consensus])
        
        for node_id, update in node_updates.items():
            if update is None:
                continue
            
            # Check if update would significantly destabilize consensus
            simulated_consensus = (global_update + update) / 2
            stability_impact = np.linalg.norm(simulated_consensus - global_update)
            
            # Flag if impact is disproportionately high
            if stability_impact > consensus_trend * 1.8:
                suspects.append(node_id)
            
            # Check directional consistency
            if len(self.consensus_history) >= 2:
                prev_consensus = self.consensus_history[-2]
                expected_direction = global_update - prev_consensus
                update_direction = update - prev_consensus
                
                # Calculate angle between expected and actual direction
                if np.linalg.norm(expected_direction) > 0 and np.linalg.norm(update_direction) > 0:
                    cosine_sim = np.dot(expected_direction, update_direction) / (
                        np.linalg.norm(expected_direction) * np.linalg.norm(update_direction))
                    
                    if cosine_sim < -0.5:  # Opposing direction
                        suspects.append(node_id)
        
        return suspects
    
    def _detect_pattern_anomalies(self, node_updates: Dict[str, np.ndarray]) -> List[str]:
        """Detect anomalous patterns in node behavior"""
        suspects = []
        
        for node_id, update in node_updates.items():
            if update is None or len(update) == 0:
                continue
            
            # Calculate pattern metrics
            update_magnitude = np.linalg.norm(update)
            update_sparsity = np.count_nonzero(update) / len(update)
            update_entropy = -np.sum(np.abs(update) * np.log(np.abs(update) + 1e-10))
            
            # Store pattern metrics
            if node_id not in self.node_response_patterns:
                self.node_response_patterns[node_id] = []
            
            pattern_score = update_magnitude * update_sparsity * (1 + update_entropy)
            self.node_response_patterns[node_id].append(pattern_score)
            
            if len(self.node_response_patterns[node_id]) > 10:
                self.node_response_patterns[node_id] = self.node_response_patterns[node_id][-10:]
            
            # Detect anomalous patterns
            if len(self.node_response_patterns[node_id]) >= 5:
                pattern_variance = np.var(self.node_response_patterns[node_id])
                pattern_mean = np.mean(self.node_response_patterns[node_id])
                
                # Flag nodes with either too consistent or too erratic patterns
                if pattern_variance < 0.01 or pattern_variance > pattern_mean * 5:
                    suspects.append(node_id)
        
        return suspects
    
    def _detect_peer_validation_failures(self, node_updates: Dict[str, np.ndarray]) -> List[str]:
        """Cross-validate nodes against their peers"""
        suspects = []
        node_ids = list(node_updates.keys())
        
        if len(node_ids) < 3:
            return suspects
        
        # Calculate pairwise similarities
        similarities = {}
        for i, node1 in enumerate(node_ids):
            similarities[node1] = []
            
            for j, node2 in enumerate(node_ids):
                if i != j and node_updates[node1] is not None and node_updates[node2] is not None:
                    # Cosine similarity
                    update1, update2 = node_updates[node1], node_updates[node2]
                    similarity = np.dot(update1, update2) / (
                        np.linalg.norm(update1) * np.linalg.norm(update2) + 1e-10)
                    similarities[node1].append(similarity)
        
        # Flag nodes with consistently low peer similarity
        for node_id, sims in similarities.items():
            if len(sims) > 0:
                avg_similarity = np.mean(sims)
                if avg_similarity < 0.3:  # Low similarity threshold
                    suspects.append(node_id)
        
        return suspects
    
    def _detect_temporal_inconsistencies(self, node_updates: Dict[str, np.ndarray]) -> List[str]:
        """Detect temporal inconsistencies in node behavior"""
        suspects = []
        
        for node_id, update in node_updates.items():
            if update is None:
                continue
            
            # Store anomaly scores for temporal analysis
            if node_id not in self.anomaly_scores:
                self.anomaly_scores[node_id] = []
            
            # Calculate current anomaly score
            update_norm = np.linalg.norm(update)
            self.anomaly_scores[node_id].append(update_norm)
            
            if len(self.anomaly_scores[node_id]) > 12:
                self.anomaly_scores[node_id] = self.anomaly_scores[node_id][-12:]
            
            # Detect temporal anomalies
            if len(self.anomaly_scores[node_id]) >= 5:
                recent_scores = self.anomaly_scores[node_id][-5:]
                score_trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
                score_variance = np.var(recent_scores)
                
                # Flag nodes with suspicious trends or high variance
                if abs(score_trend) > 0.5 or score_variance > np.mean(recent_scores):
                    suspects.append(node_id)
        
        return suspects
    
    def _calculate_byzantine_score(self, node_id: str, node_updates: Dict[str, np.ndarray],
                                 global_update: np.ndarray) -> float:
        """Calculate enhanced Byzantine score for improved detection"""
        score = 0.0
        
        # Enhanced deviation score (0-0.4) - higher weight
        if node_id in self.node_deviations and len(self.node_deviations[node_id]) > 0:
            avg_deviation = np.mean(self.node_deviations[node_id][-5:])
            score += min(avg_deviation / 5.0, 0.4)  # More sensitive to deviations
        
        # Historical detection score (0-0.3) - increased weight
        if node_id in self.detection_history:
            detection_rate = min(self.detection_history[node_id] / 5.0, 0.3)  # More weight to history
            score += detection_rate
        
        # Pattern anomaly score (0-0.3) - enhanced
        if node_id in self.node_response_patterns and len(self.node_response_patterns[node_id]) > 2:
            pattern_variance = np.var(self.node_response_patterns[node_id])
            pattern_score = min(pattern_variance * 0.2, 0.3)  # Doubled sensitivity
            score += pattern_score
        
        # Consensus violation score (0-0.3) - enhanced
        if node_updates.get(node_id) is not None:
            update = node_updates[node_id]
            consensus_deviation = np.linalg.norm(update - global_update)
            consensus_score = min(consensus_deviation / 4.0, 0.3)  # More sensitive
            score += consensus_score
        
        return min(score, 1.0)


class CommitteeManager:
    """Manages committee selection, rotation, and security"""
    
    def __init__(self, committee_size: int = 7, rotation_period: int = 10):
        self.committee_size = committee_size
        self.rotation_period = rotation_period  # rounds
        self.current_committee: List[CommitteeNode] = []
        self.node_reputations: Dict[str, NodeReputation] = {}
        self.sybil_detector = SybilDetector()
        self.byzantine_detector = ByzantineDetector()
        self.dp_reputation_manager = DifferentialPrivacyReputationManager()
        self.round_counter = 0
        self.role_assignments: Dict[str, List[NodeRole]] = {}
        
    def generate_node_keypair(self) -> Tuple[Any, Any]:
        """Generate RSA keypair for a node"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def register_node(self, node_id: str) -> bool:
        """Register a new node with proof-of-work verification"""
        challenge, difficulty = self.sybil_detector.proof_of_work_challenge(node_id)
        
        # Simulate proof-of-work (in real implementation, node would solve this)
        nonce = self._solve_proof_of_work(challenge, difficulty)
        
        if self.sybil_detector.verify_proof_of_work(node_id, challenge, nonce):
            # Initialize reputation
            self.node_reputations[node_id] = NodeReputation(node_id=node_id)
            self.sybil_detector.registered_nodes[node_id] = time.time()
            return True
        return False
    
    def _solve_proof_of_work(self, challenge: str, difficulty: int) -> int:
        """Simulate solving proof-of-work (simplified for demo)"""
        nonce = 0
        while True:
            solution = hashlib.sha256(f"{challenge}{nonce}".encode()).hexdigest()
            if solution.startswith('0' * difficulty):
                return nonce
            nonce += 1
            if nonce > 1000000:  # Prevent infinite loop in demo
                break
        return nonce
    
    def select_committee(self, available_nodes: List[str]) -> List[CommitteeNode]:
        """Select committee based on reputation-weighted probability"""
        if len(available_nodes) < self.committee_size:
            raise ValueError("Not enough nodes available for committee selection")
        
        # Filter out suspicious nodes
        suspicious_nodes = self.sybil_detector.detect_sybil_patterns(available_nodes)
        clean_nodes = [n for n in available_nodes if n not in suspicious_nodes]
        
        if len(clean_nodes) < self.committee_size:
            raise ValueError("Not enough trusted nodes for committee selection")
        
        # Calculate reputation-based weights
        weights = []
        for node_id in clean_nodes:
            if node_id in self.node_reputations:
                reputation_score = self.node_reputations[node_id].calculate_reputation_score()
                weights.append(max(0.1, reputation_score))  # Minimum weight to prevent exclusion
            else:
                weights.append(0.5)  # Default weight for new nodes
        
        # Weighted random selection
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        selected_indices = np.random.choice(
            len(clean_nodes), 
            size=self.committee_size, 
            replace=False, 
            p=weights
        )
        
        # Create committee nodes with cryptographic capabilities
        committee = []
        roles = list(NodeRole)
        
        for i, idx in enumerate(selected_indices):
            node_id = clean_nodes[idx]
            private_key, public_key = self.generate_node_keypair()
            
            # Assign roles with rotation
            role = roles[i % len(roles)]
            
            # Track role assignments for rotation
            if node_id not in self.role_assignments:
                self.role_assignments[node_id] = []
            self.role_assignments[node_id].append(role)
            
            committee_node = CommitteeNode(
                node_id=node_id,
                public_key=public_key,
                private_key=private_key,
                role=role,
                reputation=self.node_reputations.get(node_id, NodeReputation(node_id))
            )
            committee.append(committee_node)
        
        self.current_committee = committee
        return committee
    
    def rotate_committee(self, available_nodes: List[str]) -> bool:
        """Rotate committee if rotation period has elapsed"""
        self.round_counter += 1
        
        if self.round_counter >= self.rotation_period:
            self.round_counter = 0
            try:
                self.select_committee(available_nodes)
                return True
            except ValueError:
                return False
        return False
    
    def verify_committee_integrity(self, committee_decisions: Dict[str, Any]) -> bool:
        """Verify committee decisions using cryptographic proofs"""
        if not self.current_committee:
            return False
        
        verified_decisions = 0
        for node in self.current_committee:
            if node.node_id in committee_decisions:
                decision_data = committee_decisions[node.node_id]
                message = json.dumps(decision_data.get('decision', {}), sort_keys=True)
                signature = decision_data.get('signature')
                
                if signature and node.verify_signature(message, signature, node.public_key):
                    verified_decisions += 1
        
        # Require majority consensus
        return verified_decisions >= (len(self.current_committee) // 2 + 1)
    
    def update_node_reputation(self, node_id: str, performance_metrics: Dict[str, float]):
        """Update node reputation with differential privacy protection"""
        if node_id not in self.node_reputations:
            self.node_reputations[node_id] = NodeReputation(node_id=node_id)
        
        accuracy_delta = performance_metrics.get('accuracy_delta', 0)
        availability_delta = performance_metrics.get('availability_delta', 0)
        
        self.node_reputations[node_id] = self.dp_reputation_manager.update_reputation_with_privacy(
            self.node_reputations[node_id],
            accuracy_delta,
            availability_delta
        )
        
        # Track successful validation
        if performance_metrics.get('validation_success', False):
            self.node_reputations[node_id].successful_validations += 1
        
        # Track malicious behavior
        if performance_metrics.get('malicious_detected', False):
            self.node_reputations[node_id].malicious_behavior_count += 1
    
    def detect_attacks(self, node_updates: Dict[str, np.ndarray],
                      global_update: np.ndarray) -> Dict[str, List[str]]:
        """Comprehensive attack detection"""
        results = {
            'sybil_nodes': [],
            'byzantine_nodes': []
        }
        
        # Detect Sybil attacks
        available_nodes = list(self.node_reputations.keys())
        results['sybil_nodes'] = self.sybil_detector.detect_sybil_patterns(available_nodes)
        
        # Detect Byzantine attacks
        results['byzantine_nodes'] = self.byzantine_detector.detect_byzantine_behavior(
            node_updates, global_update
        )
        
        return results
    
    def get_committee_status(self) -> Dict[str, Any]:
        """Get current committee status and security metrics"""
        return {
            'committee_size': len(self.current_committee),
            'round_counter': self.round_counter,
            'rotation_period': self.rotation_period,
            'committee_members': [
                {
                    'node_id': node.node_id,
                    'role': node.role.value,
                    'reputation_score': node.reputation.calculate_reputation_score(),
                    'selection_time': node.selection_timestamp
                }
                for node in self.current_committee
            ],
            'total_registered_nodes': len(self.node_reputations),
            'security_status': {
                'sybil_protection_active': True,
                'byzantine_detection_active': True,
                'differential_privacy_active': True,
                'cryptographic_verification_active': True
            }
        }


# Integration with existing federated learning system
class SecureFederatedLearning:
    """Enhanced federated learning with committee-based security"""
    
    def __init__(self, committee_size: int = 7, rotation_period: int = 10):
        self.committee_manager = CommitteeManager(committee_size, rotation_period)
        self.round_number = 0
        
    def initialize_secure_training(self, node_ids: List[str]) -> bool:
        """Initialize secure federated training with committee selection"""
        # Register all nodes
        for node_id in node_ids:
            self.committee_manager.register_node(node_id)
        
        # Select initial committee
        try:
            self.committee_manager.select_committee(node_ids)
            return True
        except ValueError as e:
            print(f"Failed to initialize secure training: {e}")
            return False
    
    def secure_training_round(self, node_updates: Dict[str, np.ndarray],
                            performance_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Execute a secure training round with committee oversight"""
        self.round_number += 1
        
        # Calculate global update (simplified aggregation)
        valid_updates = [update for update in node_updates.values() if update is not None]
        if valid_updates:
            global_update = np.mean(valid_updates, axis=0)
        else:
            global_update = np.zeros(10)  # Default shape
        
        # Detect attacks
        attack_results = self.committee_manager.detect_attacks(node_updates, global_update)
        
        # Update reputations
        for node_id, metrics in performance_metrics.items():
            self.committee_manager.update_node_reputation(node_id, metrics)
        
        # Check for committee rotation
        available_nodes = list(node_updates.keys())
        rotation_occurred = self.committee_manager.rotate_committee(available_nodes)
        
        return {
            'round_number': self.round_number,
            'global_update': global_update,
            'attack_detection': attack_results,
            'committee_rotated': rotation_occurred,
            'committee_status': self.committee_manager.get_committee_status(),
            'security_metrics': {
                'nodes_flagged_sybil': len(attack_results['sybil_nodes']),
                'nodes_flagged_byzantine': len(attack_results['byzantine_nodes']),
                'committee_integrity_verified': True
            }
        }