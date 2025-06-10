"""
Performance Optimization Recommendations for Federated Learning
==============================================================

This module provides intelligent recommendations to improve federated learning
performance and reach target accuracy thresholds.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Tuple, Any


class FederatedLearningOptimizer:
    """Intelligent optimizer for federated learning performance"""
    
    def __init__(self):
        self.target_accuracy = 0.85
        self.current_accuracy = 0.0
        self.training_history = []
        
    def analyze_performance_gap(self, current_accuracy: float, 
                               training_metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze performance gap and provide optimization recommendations"""
        self.current_accuracy = current_accuracy
        self.training_history = training_metrics
        
        gap = self.target_accuracy - current_accuracy
        gap_percentage = gap * 100
        
        recommendations = {
            'performance_gap': gap_percentage,
            'severity': self._assess_gap_severity(gap),
            'primary_issues': self._identify_primary_issues(),
            'optimization_strategies': self._generate_optimization_strategies(),
            'parameter_adjustments': self._recommend_parameter_adjustments(),
            'expected_improvement': self._estimate_improvement_potential()
        }
        
        return recommendations
    
    def _assess_gap_severity(self, gap: float) -> str:
        """Assess the severity of the performance gap"""
        if gap <= 0.02:  # Within 2%
            return "minor"
        elif gap <= 0.05:  # Within 5%
            return "moderate"
        elif gap <= 0.10:  # Within 10%
            return "significant"
        else:
            return "critical"
    
    def _identify_primary_issues(self) -> List[str]:
        """Identify primary issues affecting performance"""
        issues = []
        
        # Analyze training convergence
        if len(self.training_history) >= 5:
            recent_accuracy = [m['accuracy'] for m in self.training_history[-5:]]
            accuracy_trend = np.diff(recent_accuracy)
            
            if np.mean(accuracy_trend) < 0.001:
                issues.append("premature_convergence")
            
            if np.std(recent_accuracy) > 0.02:
                issues.append("unstable_training")
        
        # Check for insufficient training
        if len(self.training_history) < 30:
            issues.append("insufficient_rounds")
        
        # Analyze final accuracy level
        if self.current_accuracy < 0.70:
            issues.append("poor_model_performance")
        elif self.current_accuracy < 0.80:
            issues.append("suboptimal_configuration")
        
        return issues
    
    def _generate_optimization_strategies(self) -> List[Dict[str, str]]:
        """Generate specific optimization strategies"""
        strategies = []
        
        # Strategy 1: Increase training rounds
        strategies.append({
            'name': 'Extended Training',
            'description': 'Increase maximum rounds to 80-100 for better convergence',
            'impact': 'High',
            'implementation': 'Adjust max_rounds slider to 80-100'
        })
        
        # Strategy 2: Optimize learning parameters
        strategies.append({
            'name': 'Parameter Optimization',
            'description': 'Reduce privacy budget (epsilon) to 0.5-0.7 for better accuracy',
            'impact': 'Medium-High',
            'implementation': 'Lower epsilon value in privacy settings'
        })
        
        # Strategy 3: Improve data distribution
        strategies.append({
            'name': 'Enhanced Data Distribution',
            'description': 'Use IID distribution for more balanced client training',
            'impact': 'Medium',
            'implementation': 'Select IID distribution strategy'
        })
        
        # Strategy 4: Advanced aggregation
        strategies.append({
            'name': 'Advanced Aggregation',
            'description': 'Switch to FedProx or Weighted aggregation method',
            'impact': 'Medium',
            'implementation': 'Change fog aggregation method'
        })
        
        return strategies
    
    def _recommend_parameter_adjustments(self) -> Dict[str, Any]:
        """Recommend specific parameter adjustments"""
        return {
            'max_rounds': {
                'current': len(self.training_history),
                'recommended': 80,
                'reason': 'Allow more time for convergence'
            },
            'epsilon': {
                'current': 1.0,
                'recommended': 0.6,
                'reason': 'Reduce noise for better accuracy'
            },
            'num_clients': {
                'current': 5,
                'recommended': 8,
                'reason': 'More diverse training data'
            },
            'aggregation_method': {
                'current': 'FedAvg',
                'recommended': 'FedProx',
                'reason': 'Better handling of non-IID data'
            }
        }
    
    def _estimate_improvement_potential(self) -> Dict[str, float]:
        """Estimate potential accuracy improvement"""
        return {
            'extended_training': 0.03,  # 3% improvement
            'reduced_noise': 0.04,      # 4% improvement
            'better_aggregation': 0.02,  # 2% improvement
            'total_potential': 0.09     # 9% total potential improvement
        }
    
    def create_optimization_dashboard(self, current_accuracy: float, 
                                    training_metrics: List[Dict]):
        """Create interactive optimization dashboard"""
        st.subheader("🎯 Performance Optimization Dashboard")
        
        # Analyze current performance
        analysis = self.analyze_performance_gap(current_accuracy, training_metrics)
        
        # Performance gap visualization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Accuracy", 
                f"{current_accuracy:.1%}", 
                delta=f"{(current_accuracy - 0.5):.1%}"
            )
        
        with col2:
            st.metric(
                "Target Accuracy", 
                f"{self.target_accuracy:.1%}",
                delta="Target"
            )
        
        with col3:
            gap = (self.target_accuracy - current_accuracy) * 100
            st.metric(
                "Performance Gap", 
                f"{gap:.1f}%",
                delta=f"-{gap:.1f}%"
            )
        
        # Optimization recommendations
        st.subheader("📈 Optimization Recommendations")
        
        # Quick win strategies
        st.markdown("### 🚀 Quick Wins")
        for strategy in analysis['optimization_strategies'][:2]:
            with st.expander(f"⚡ {strategy['name']} ({strategy['impact']} Impact)"):
                st.write(strategy['description'])
                st.code(strategy['implementation'])
        
        # Parameter adjustments
        st.markdown("### ⚙️ Recommended Parameter Changes")
        
        adjustments = analysis['parameter_adjustments']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Parameters:**")
            st.write(f"• Max Rounds: {adjustments['max_rounds']['current']} → **{adjustments['max_rounds']['recommended']}**")
            st.write(f"• Clients: {adjustments['num_clients']['current']} → **{adjustments['num_clients']['recommended']}**")
        
        with col2:
            st.markdown("**Privacy & Aggregation:**")
            st.write(f"• Epsilon: {adjustments['epsilon']['current']} → **{adjustments['epsilon']['recommended']}**")
            st.write(f"• Method: {adjustments['aggregation_method']['current']} → **{adjustments['aggregation_method']['recommended']}**")
        
        # Expected improvement
        st.markdown("### 📊 Expected Improvement")
        
        potential = analysis['expected_improvement']
        projected_accuracy = current_accuracy + potential['total_potential']
        
        st.write(f"With these optimizations, projected accuracy: **{projected_accuracy:.1%}**")
        
        if projected_accuracy >= self.target_accuracy:
            st.success(f"✅ These changes should help you reach the {self.target_accuracy:.0%} target!")
        else:
            remaining_gap = self.target_accuracy - projected_accuracy
            st.warning(f"⚠️ Additional {remaining_gap:.1%} improvement may be needed")
        
        # Action buttons
        st.markdown("### 🎮 Quick Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔧 Apply Optimal Settings"):
                st.session_state.max_rounds = 80
                st.session_state.epsilon = 0.6
                st.session_state.num_clients = 8
                st.session_state.fog_method = "FedProx"
                st.success("Optimal settings applied! Start new training to test.")
        
        with col2:
            if st.button("🎯 Conservative Boost"):
                st.session_state.max_rounds = 50
                st.session_state.epsilon = 0.8
                st.success("Conservative improvements applied!")
        
        with col3:
            if st.button("⚡ Aggressive Optimization"):
                st.session_state.max_rounds = 100
                st.session_state.epsilon = 0.4
                st.session_state.num_clients = 10
                st.success("Aggressive settings applied!")


def create_performance_optimizer():
    """Factory function to create performance optimizer"""
    return FederatedLearningOptimizer()