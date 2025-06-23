import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from typing import Dict, List, Any
import seaborn as sns
import matplotlib.pyplot as plt

class ClientPerformanceVisualizer:
    """Comprehensive visualization for client performance tracking across rounds"""
    
    def __init__(self):
        self.client_metrics_history = {}
        self.round_confusion_matrices = {}
        self.client_accuracy_trends = {}
        
    def update_client_metrics(self, round_num: int, client_id: int, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
        """Update metrics for a specific client in a specific round"""
        if round_num not in self.client_metrics_history:
            self.client_metrics_history[round_num] = {}
        
        if round_num not in self.round_confusion_matrices:
            self.round_confusion_matrices[round_num] = {}
            
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)
        
        # Store metrics
        self.client_metrics_history[round_num][client_id] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred,
            'true_labels': y_true,
            'probabilities': y_prob if y_prob is not None else np.ones(len(y_true)) * 0.5
        }
        
        self.round_confusion_matrices[round_num][client_id] = cm
        
        # Update accuracy trends
        if client_id not in self.client_accuracy_trends:
            self.client_accuracy_trends[client_id] = {'rounds': [], 'accuracy': [], 'f1_score': []}
        
        self.client_accuracy_trends[client_id]['rounds'].append(round_num)
        self.client_accuracy_trends[client_id]['accuracy'].append(accuracy)
        self.client_accuracy_trends[client_id]['f1_score'].append(f1)
    
    def create_client_accuracy_dashboard(self):
        """Create comprehensive client accuracy dashboard"""
        if not self.client_metrics_history:
            st.warning("No client performance data available yet.")
            return
        
        st.subheader("📊 Client Performance Dashboard")
        
        # Overall performance overview
        col1, col2 = st.columns(2)
        
        with col1:
            self._create_accuracy_trends_plot()
        
        with col2:
            self._create_client_comparison_radar()
        
        # Round-by-round analysis
        st.subheader("🔍 Round-by-Round Analysis")
        
        # Round selector
        available_rounds = sorted(list(self.client_metrics_history.keys()))
        if available_rounds:
            selected_round = st.selectbox("Select Round for Detailed Analysis", available_rounds)
            
            self._create_round_analysis(selected_round)
    
    def _create_accuracy_trends_plot(self):
        """Create accuracy trends plot for all clients"""
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, (client_id, trends) in enumerate(self.client_accuracy_trends.items()):
            if trends['rounds']:  # Only plot if data exists
                color = colors[i % len(colors)]
                
                # Accuracy line
                fig.add_trace(go.Scatter(
                    x=trends['rounds'],
                    y=trends['accuracy'],
                    mode='lines+markers',
                    name=f'Client {client_id}',
                    line=dict(color=color, width=3),
                    marker=dict(size=8, symbol='circle')
                ))
        
        fig.update_layout(
            title="Client Accuracy Trends Across Rounds",
            xaxis_title="Training Round",
            yaxis_title="Accuracy",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(range=[0, 1])
        
        st.plotly_chart(fig, use_container_width=True, key="client_accuracy_trends")
    
    def _create_client_comparison_radar(self):
        """Create radar chart comparing client performances"""
        if not self.client_metrics_history:
            return
        
        # Get latest round data
        latest_round = max(self.client_metrics_history.keys())
        round_data = self.client_metrics_history[latest_round]
        
        if not round_data:
            return
        
        # Prepare data for radar chart
        clients = list(round_data.keys())
        metrics = ['Accuracy', 'F1 Score', 'Data Quality', 'Convergence']
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, client_id in enumerate(clients):
            client_data = round_data[client_id]
            
            # Calculate synthetic metrics for demonstration
            data_quality = np.random.uniform(0.7, 1.0)  # Mock data quality score
            convergence = min(1.0, client_data['accuracy'] + 0.1)  # Mock convergence score
            
            values = [
                client_data['accuracy'],
                client_data['f1_score'],
                data_quality,
                convergence
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # Close the polygon
                theta=metrics + [metrics[0]],
                fill='toself',
                name=f'Client {client_id}',
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            title=f"Client Performance Comparison (Round {latest_round})",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key="client_performance_radar")
    
    def _create_round_analysis(self, round_num: int):
        """Create detailed analysis for a specific round"""
        if round_num not in self.client_metrics_history:
            st.warning(f"No data available for round {round_num}")
            return
        
        round_data = self.client_metrics_history[round_num]
        round_cms = self.round_confusion_matrices.get(round_num, {})
        
        # Performance metrics table
        st.subheader(f"📈 Round {round_num} Performance Metrics")
        
        metrics_data = []
        for client_id, data in round_data.items():
            metrics_data.append({
                'Client ID': f'Client {client_id}',
                'Accuracy': f"{data['accuracy']:.3f}",
                'F1 Score': f"{data['f1_score']:.3f}",
                'Samples': len(data['true_labels']),
                'Positive Predictions': np.sum(data['predictions']),
                'True Positives': np.sum((data['true_labels'] == 1) & (data['predictions'] == 1))
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Confusion matrices
        st.subheader(f"🎯 Confusion Matrices - Round {round_num}")
        
        # Create confusion matrix plots
        if round_cms:
            cols = st.columns(min(3, len(round_cms)))
            
            for idx, (client_id, cm) in enumerate(round_cms.items()):
                with cols[idx % len(cols)]:
                    self._create_confusion_matrix_plot(cm, f"Client {client_id}")
        
        # Client distribution analysis
        st.subheader(f"📊 Prediction Distribution Analysis - Round {round_num}")
        self._create_prediction_distribution_plot(round_data)
    
    def _create_confusion_matrix_plot(self, cm: np.ndarray, title: str):
        """Create confusion matrix heatmap"""
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'])
        
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        st.pyplot(fig)
        plt.close(fig)
    
    def _create_prediction_distribution_plot(self, round_data: Dict):
        """Create prediction distribution plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy Distribution', 'F1 Score Distribution', 
                           'Prediction Confidence', 'Sample Size Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'histogram'}],
                   [{'type': 'box'}, {'type': 'bar'}]]
        )
        
        client_ids = list(round_data.keys())
        accuracies = [round_data[cid]['accuracy'] for cid in client_ids]
        f1_scores = [round_data[cid]['f1_score'] for cid in client_ids]
        sample_sizes = [len(round_data[cid]['true_labels']) for cid in client_ids]
        
        # Accuracy distribution
        fig.add_trace(
            go.Histogram(x=accuracies, nbinsx=10, name='Accuracy', showlegend=False),
            row=1, col=1
        )
        
        # F1 Score distribution
        fig.add_trace(
            go.Histogram(x=f1_scores, nbinsx=10, name='F1 Score', showlegend=False),
            row=1, col=2
        )
        
        # Prediction confidence (using probabilities)
        all_probabilities = []
        for client_data in round_data.values():
            all_probabilities.extend(client_data['probabilities'])
        
        fig.add_trace(
            go.Box(y=all_probabilities, name='Confidence', showlegend=False),
            row=2, col=1
        )
        
        # Sample size distribution
        fig.add_trace(
            go.Bar(x=[f'Client {cid}' for cid in client_ids], y=sample_sizes, 
                   name='Sample Size', showlegend=False),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Round Performance Distribution Analysis")
        st.plotly_chart(fig, use_container_width=True, key=f"round_analysis_{hash(str(round_data))}")
    
    def create_global_performance_summary(self):
        """Create global performance summary across all rounds and clients"""
        if not self.client_metrics_history:
            st.warning("No performance data available.")
            return
        
        st.subheader("🌍 Global Performance Summary")
        
        # Aggregate statistics
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate global stats
        all_accuracies = []
        all_f1_scores = []
        total_clients = set()
        total_rounds = len(self.client_metrics_history)
        
        for round_data in self.client_metrics_history.values():
            for client_id, metrics in round_data.items():
                all_accuracies.append(metrics['accuracy'])
                all_f1_scores.append(metrics['f1_score'])
                total_clients.add(client_id)
        
        with col1:
            st.metric("Average Accuracy", f"{np.mean(all_accuracies):.3f}")
        
        with col2:
            st.metric("Average F1 Score", f"{np.mean(all_f1_scores):.3f}")
        
        with col3:
            st.metric("Total Clients", len(total_clients))
        
        with col4:
            st.metric("Total Rounds", total_rounds)
        
        # Performance evolution heatmap
        self._create_performance_heatmap()
    
    def _create_performance_heatmap(self):
        """Create heatmap showing performance evolution"""
        if not self.client_metrics_history:
            return
        
        # Prepare data matrix
        rounds = sorted(self.client_metrics_history.keys())
        all_clients = set()
        for round_data in self.client_metrics_history.values():
            all_clients.update(round_data.keys())
        clients = sorted(all_clients)
        
        # Create accuracy matrix
        accuracy_matrix = []
        for client_id in clients:
            client_accuracies = []
            for round_num in rounds:
                if (round_num in self.client_metrics_history and 
                    client_id in self.client_metrics_history[round_num]):
                    accuracy = self.client_metrics_history[round_num][client_id]['accuracy']
                    client_accuracies.append(accuracy)
                else:
                    client_accuracies.append(np.nan)
            accuracy_matrix.append(client_accuracies)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=accuracy_matrix,
            x=[f'Round {r}' for r in rounds],
            y=[f'Client {c}' for c in clients],
            colorscale='RdYlBu_r',
            zmid=0.5,
            zmin=0,
            zmax=1,
            colorbar=dict(title="Accuracy")
        ))
        
        fig.update_layout(
            title="Client Performance Evolution Heatmap",
            xaxis_title="Training Rounds",
            yaxis_title="Clients",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True, key="performance_heatmap_summary")

def simulate_client_performance_data(visualizer: ClientPerformanceVisualizer, num_clients: int = 5, num_rounds: int = 10):
    """Simulate client performance data for demonstration"""
    np.random.seed(42)
    
    for round_num in range(1, num_rounds + 1):
        for client_id in range(num_clients):
            # Simulate improving performance over rounds
            base_accuracy = 0.6 + (round_num - 1) * 0.02 + np.random.normal(0, 0.05)
            base_accuracy = np.clip(base_accuracy, 0.4, 0.95)
            
            # Generate synthetic test data
            n_samples = np.random.randint(20, 100)
            y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate
            
            # Generate predictions based on accuracy
            y_pred = y_true.copy()
            errors = np.random.choice(n_samples, int(n_samples * (1 - base_accuracy)), replace=False)
            y_pred[errors] = 1 - y_pred[errors]
            
            # Generate probabilities
            y_prob = np.random.uniform(0.1, 0.9, n_samples)
            
            visualizer.update_client_metrics(round_num, client_id, y_true, y_pred, y_prob)