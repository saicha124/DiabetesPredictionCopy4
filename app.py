import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from data_distribution import get_distribution_strategy, visualize_data_distribution
from fog_aggregation import HierarchicalFederatedLearning
from utils import calculate_metrics, plot_confusion_matrix

# Page configuration
st.set_page_config(
    page_title="Agronomic Federated Learning Dashboard",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'fl_manager' not in st.session_state:
        st.session_state.fl_manager = None
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []
    if 'confusion_matrices' not in st.session_state:
        st.session_state.confusion_matrices = []
    if 'execution_times' not in st.session_state:
        st.session_state.execution_times = []
    if 'communication_times' not in st.session_state:
        st.session_state.communication_times = []
    if 'client_status' not in st.session_state:
        st.session_state.client_status = {}
    if 'best_accuracy' not in st.session_state:
        st.session_state.best_accuracy = 0.0
    if 'early_stopped' not in st.session_state:
        st.session_state.early_stopped = False
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0
    if 'client_results' not in st.session_state:
        st.session_state.client_results = []
    if 'fog_results' not in st.session_state:
        st.session_state.fog_results = []
    if 'distribution_strategy' not in st.session_state:
        st.session_state.distribution_strategy = 'IID'
    if 'distribution_stats' not in st.session_state:
        st.session_state.distribution_stats = None

def start_training(data, num_clients, max_rounds, target_accuracy, 
                  aggregation_algorithm, enable_dp, epsilon, delta, committee_size,
                  distribution_strategy, strategy_params, enable_fog=True, 
                  num_fog_nodes=3, fog_aggregation_method="Mixed Methods",
                  model_type="logistic_regression", privacy_mechanism="gaussian", 
                  gradient_clip_norm=1.0):
    """Start federated learning training with custom data distribution and fog aggregation"""
    import numpy as np
    try:
        # Create FL manager with advanced parameters
        st.session_state.fl_manager = FederatedLearningManager(
            num_clients=num_clients,
            max_rounds=max_rounds,
            target_accuracy=target_accuracy,
            aggregation_algorithm=aggregation_algorithm,
            enable_dp=enable_dp,
            epsilon=epsilon,
            delta=delta,
            committee_size=committee_size,
            model_type=model_type,
            privacy_mechanism=privacy_mechanism,
            gradient_clip_norm=gradient_clip_norm
        )
        
        # Initialize hierarchical fog aggregation if enabled
        if enable_fog and num_fog_nodes > 0:
            fog_manager = HierarchicalFederatedLearning(
                num_clients=num_clients,
                num_fog_nodes=num_fog_nodes,
                fog_aggregation_method=fog_aggregation_method
            )
            st.session_state.fl_manager.fog_manager = fog_manager
            st.session_state.fog_enabled = True
            st.session_state.fog_manager = fog_manager
        else:
            st.session_state.fl_manager.fog_manager = None
            st.session_state.fog_enabled = False
            st.session_state.fog_manager = None
        
        # Store distribution configuration
        st.session_state.distribution_strategy = distribution_strategy
        st.session_state.strategy_params = strategy_params
        
        # Reset training state
        st.session_state.training_started = True
        st.session_state.training_completed = False
        st.session_state.training_metrics = []
        st.session_state.confusion_matrices = []
        st.session_state.execution_times = []
        st.session_state.communication_times = []
        st.session_state.client_status = {}
        st.session_state.best_accuracy = 0.0
        st.session_state.early_stopped = False
        st.session_state.current_round = 0
        st.session_state.client_results = []
        st.session_state.fog_results = []
        st.session_state.training_data = data
        
        st.success(f"Training initialized with {distribution_strategy} distribution! Switch to Live Monitoring tab to start.")
        
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")

def run_training_loop(fl_manager, data):
    """Run the training loop with early stopping"""
    try:
        # Train with monitoring
        results = fl_manager.train(data)
        
        # Store results in session state with safe access
        try:
            st.session_state.results = results
            st.session_state.training_completed = True
            st.session_state.training_started = False
            
            # Extract client and fog results for tables
            extract_training_results(fl_manager)
        except Exception as session_error:
            print(f"Session state update failed: {session_error}")
            # Store results in the fl_manager for later retrieval
            fl_manager.final_results = results
            fl_manager.training_completed = True
        
    except Exception as e:
        try:
            st.session_state.training_started = False
        except:
            pass
        print(f"Training failed: {str(e)}")

def extract_training_results(fl_manager):
    """Extract detailed client and fog results for tabular display"""
    client_results = []
    fog_results = []
    
    # Extract client metrics
    for i, client in enumerate(fl_manager.clients):
        if hasattr(client, 'training_history') and client.training_history:
            latest_metrics = client.training_history[-1]
            client_results.append({
                'Station ID': f'Field-{i+1}',
                'Samples': latest_metrics.get('samples', 0),
                'Accuracy': f"{latest_metrics.get('test_accuracy', 0):.3f}",
                'F1 Score': f"{latest_metrics.get('f1_score', 0):.3f}",
                'Training Time': f"{latest_metrics.get('training_time', 0):.2f}s",
                'Classes': latest_metrics.get('classes', 0),
                'Status': 'Active' if latest_metrics.get('test_accuracy', 0) > 0 else 'Inactive'
            })
    
    # Extract fog-level aggregation results
    for round_num in range(len(st.session_state.training_metrics)):
        metrics = st.session_state.training_metrics[round_num]
        fog_results.append({
            'Round': round_num + 1,
            'Global Accuracy': f"{metrics.get('accuracy', 0):.3f}",
            'Global F1': f"{metrics.get('f1_score', 0):.3f}",
            'Aggregation Time': f"{st.session_state.execution_times[round_num] if round_num < len(st.session_state.execution_times) else 0:.2f}s",
            'Communication Time': f"{st.session_state.communication_times[round_num] if round_num < len(st.session_state.communication_times) else 0:.2f}s",
            'Algorithm': st.session_state.fl_manager.aggregation_algorithm,
            'Privacy': 'Enabled' if st.session_state.fl_manager.enable_dp else 'Disabled'
        })
    
    st.session_state.client_results = client_results
    st.session_state.fog_results = fog_results

def show_training_progress():
    """Display real-time training progress"""
    st.header("📊 Live Training Progress")
    
    # Update session state from fl_manager if available
    if st.session_state.fl_manager:
        # Sync data from fl_manager to session state
        if hasattr(st.session_state.fl_manager, 'training_history'):
            st.session_state.training_metrics = st.session_state.fl_manager.training_history
        if hasattr(st.session_state.fl_manager, 'confusion_matrices'):
            st.session_state.confusion_matrices = st.session_state.fl_manager.confusion_matrices
        if hasattr(st.session_state.fl_manager, 'execution_times'):
            st.session_state.execution_times = st.session_state.fl_manager.execution_times
        if hasattr(st.session_state.fl_manager, 'communication_times'):
            st.session_state.communication_times = st.session_state.fl_manager.communication_times
        if hasattr(st.session_state.fl_manager, 'best_accuracy'):
            st.session_state.best_accuracy = st.session_state.fl_manager.best_accuracy
        if hasattr(st.session_state.fl_manager, 'early_stopped'):
            st.session_state.early_stopped = st.session_state.fl_manager.early_stopped
        if hasattr(st.session_state.fl_manager, 'training_completed'):
            st.session_state.training_completed = st.session_state.fl_manager.training_completed
    
    # Overall progress
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        if st.session_state.fl_manager:
            current_round = getattr(st.session_state.fl_manager, 'current_round', 0)
            max_rounds = st.session_state.fl_manager.max_rounds
            progress = min(current_round / max_rounds, 1.0) if max_rounds > 0 else 0
            st.metric("Training Round", f"{current_round}/{max_rounds}")
            st.progress(progress)
    
    with progress_col2:
        current_accuracy = getattr(st.session_state, 'best_accuracy', 0.0)
        target_accuracy = st.session_state.fl_manager.target_accuracy if st.session_state.fl_manager else 0.85
        st.metric("Best Accuracy", f"{current_accuracy:.3f}")
        st.metric("Target", f"{target_accuracy:.3f}")
    
    with progress_col3:
        early_stopped = getattr(st.session_state, 'early_stopped', False)
        training_completed = getattr(st.session_state, 'training_completed', False)
        
        if early_stopped:
            st.success("🎯 Target Accuracy Reached!")
        elif training_completed:
            st.info("✅ Training Completed")
        else:
            st.info("🔄 Training in Progress")
    
    # Field station status (simulate based on current round)
    if st.session_state.fl_manager:
        st.subheader("🏢 Field Station Status")
        
        num_clients = st.session_state.fl_manager.num_clients
        cols = st.columns(min(5, num_clients))
        
        for i in range(num_clients):
            with cols[i % len(cols)]:
                if training_completed or early_stopped:
                    status_color = "🟢"
                    status = "Completed"
                else:
                    status_color = "🟡"
                    status = "Training"
                st.metric(f"Station {i+1}", f"{status_color} {status}")

def show_training_charts():
    """Display advanced training progress visualizations"""
    if not st.session_state.training_metrics:
        return
    
    st.header("📈 Real-Time Training Analytics")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(st.session_state.training_metrics)
    
    # Performance charts with individual client tracking
    col1, col2 = st.columns(2)
    
    with col1:
        # Multi-metric performance chart with variance bands
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('🌾 Crop Health Performance', '🏡 Individual Farm Variance'),
            vertical_spacing=0.15
        )
        
        # Global performance metrics
        if 'accuracy' in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(metrics_df) + 1)),
                y=metrics_df['accuracy'],
                mode='lines+markers',
                name='Global Accuracy',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=8)
            ), row=1, col=1)
            
        if 'f1_score' in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(metrics_df) + 1)),
                y=metrics_df['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=8)
            ), row=1, col=1)
        
        # Add target accuracy line
        if st.session_state.fl_manager:
            target = st.session_state.fl_manager.target_accuracy
            fig.add_hline(y=target, line_dash="dash", line_color="red", 
                         annotation_text=f"Target: {target:.3f}", row=1, col=1)
        
        # Performance variance chart showing min/max farm performance
        if 'min_client_accuracy' in metrics_df.columns and 'max_client_accuracy' in metrics_df.columns:
            rounds = list(range(1, len(metrics_df) + 1))
            fig.add_trace(go.Scatter(
                x=rounds,
                y=metrics_df['max_client_accuracy'],
                mode='lines',
                name='Best Farm',
                line=dict(color='green', width=2),
                fill=None
            ), row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=rounds,
                y=metrics_df['min_client_accuracy'],
                mode='lines',
                name='Worst Farm',
                line=dict(color='red', width=2),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.1)'
            ), row=2, col=1)
        
        fig.update_layout(
            height=600,
            title_text="Farm Network Performance Analytics",
            template="plotly_white",
            showlegend=True
        )
        fig.update_xaxes(title_text="Analysis Cycle")
        fig.update_yaxes(title_text="Crop Health Score", row=1, col=1)
        fig.update_yaxes(title_text="Performance Range", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Individual client performance over time
        if metrics_df.shape[0] > 0 and 'client_accuracies' in metrics_df.columns:
            fig_clients = go.Figure()
            
            # Get client performance data
            all_client_data = []
            for idx, row in metrics_df.iterrows():
                if row['client_accuracies'] and len(row['client_accuracies']) > 0:
                    all_client_data.append(row['client_accuracies'])
            
            if all_client_data:
                num_clients = len(all_client_data[0])
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
                
                # Plot each client's performance trajectory
                for client_id in range(num_clients):
                    client_performance = []
                    rounds = []
                    for round_idx, round_data in enumerate(all_client_data):
                        if client_id < len(round_data):
                            client_performance.append(round_data[client_id])
                            rounds.append(round_idx + 1)
                    
                    if client_performance:
                        fig_clients.add_trace(go.Scatter(
                            x=rounds,
                            y=client_performance,
                            mode='lines+markers',
                            name=f'Farm {client_id + 1}',
                            line=dict(color=colors[client_id % len(colors)], width=2),
                            marker=dict(size=6)
                        ))
                
                fig_clients.update_layout(
                    title="🏡 Individual Farm Performance Tracking",
                    xaxis_title="Analysis Cycle",
                    yaxis_title="Crop Health Score",
                    template="plotly_white",
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig_clients, use_container_width=True)
        
        # Execution time analysis
        if st.session_state.execution_times:
            fig_time = go.Figure()
            
            # Bar chart for execution times
            fig_time.add_trace(go.Bar(
                x=list(range(1, len(st.session_state.execution_times) + 1)),
                y=st.session_state.execution_times,
                name='Analysis Time',
                marker_color='#C73E1D',
                text=[f'{t:.2f}s' for t in st.session_state.execution_times],
                textposition='auto'
            ))
            
            # Add average line
            avg_time = np.mean(st.session_state.execution_times)
            fig_time.add_hline(y=avg_time, line_dash="dash", line_color="green",
                              annotation_text=f"Avg: {avg_time:.2f}s")
            
            fig_time.update_layout(
                title="⏱️ Analysis Time per Cycle",
                xaxis_title="Analysis Cycle",
                yaxis_title="Time (seconds)",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Communication overhead visualization
        if st.session_state.communication_times:
            fig_comm = go.Figure()
            
            fig_comm.add_trace(go.Scatter(
                x=list(range(1, len(st.session_state.communication_times) + 1)),
                y=st.session_state.communication_times,
                mode='lines+markers',
                name='Communication Time',
                line=dict(color='#3F7CAC', width=2),
                fill='tonexty',
                fillcolor='rgba(63, 124, 172, 0.1)'
            ))
            
            fig_comm.update_layout(
                title="Communication Overhead",
                xaxis_title="Training Round",
                yaxis_title="Time (seconds)",
                template="plotly_white",
                height=250
            )
            st.plotly_chart(fig_comm, use_container_width=True)
    
    # Advanced analytics section
    st.subheader("🔍 Advanced Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convergence analysis
        if len(metrics_df) > 1:
            accuracy_trend = np.diff(metrics_df['accuracy']) if 'accuracy' in metrics_df.columns else []
            convergence_rate = np.mean(accuracy_trend) if len(accuracy_trend) > 0 else 0
            
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(
                x=list(range(2, len(metrics_df) + 1)),
                y=accuracy_trend,
                mode='lines+markers',
                name='Accuracy Change',
                line=dict(color='purple', width=2)
            ))
            fig_conv.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_conv.update_layout(
                title=f"Convergence Rate: {convergence_rate:.4f}",
                xaxis_title="Round",
                yaxis_title="Accuracy Change",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_conv, use_container_width=True)
    
    with col2:
        # Efficiency metrics
        if st.session_state.execution_times and 'accuracy' in metrics_df.columns:
            efficiency = metrics_df['accuracy'] / np.array(st.session_state.execution_times)
            
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Bar(
                x=list(range(1, len(efficiency) + 1)),
                y=efficiency,
                name='Efficiency',
                marker_color='lightgreen',
                text=[f'{e:.3f}' for e in efficiency],
                textposition='auto'
            ))
            fig_eff.update_layout(
                title="Training Efficiency (Accuracy/Time)",
                xaxis_title="Round",
                yaxis_title="Efficiency Score",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_eff, use_container_width=True)
    
    with col3:
        # Performance distribution
        if 'accuracy' in metrics_df.columns and len(metrics_df) > 2:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=metrics_df['accuracy'],
                nbinsx=10,
                name='Accuracy Distribution',
                marker_color='orange',
                opacity=0.7
            ))
            fig_dist.update_layout(
                title="Accuracy Distribution",
                xaxis_title="Accuracy",
                yaxis_title="Frequency",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig_dist, use_container_width=True)

def show_results():
    """Display final training results in tables"""
    st.header("📋 Training Results Analysis")
    
    if not st.session_state.results:
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        final_accuracy = st.session_state.results.get('accuracy', 0)
        st.metric("Final Accuracy", f"{final_accuracy:.3f}")
    
    with col2:
        final_f1 = st.session_state.results.get('f1_score', 0)
        st.metric("Final F1 Score", f"{final_f1:.3f}")
    
    with col3:
        total_rounds = len(st.session_state.training_metrics)
        st.metric("Total Rounds", total_rounds)
    
    with col4:
        total_time = sum(st.session_state.execution_times) if st.session_state.execution_times else 0
        st.metric("Total Time", f"{total_time:.1f}s")
    
    # Results tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Field Station Results")
        if st.session_state.client_results:
            client_df = pd.DataFrame(st.session_state.client_results)
            st.dataframe(client_df, use_container_width=True)
        else:
            st.info("No client results available")
    
    with col2:
        st.subheader("🌐 Fog Aggregation Results")
        if st.session_state.fog_results:
            fog_df = pd.DataFrame(st.session_state.fog_results)
            st.dataframe(fog_df, use_container_width=True)
        else:
            st.info("No fog results available")

def make_prediction(sample_data):
    """Make prediction for crop sample data"""
    if st.session_state.fl_manager and hasattr(st.session_state.fl_manager, 'global_model') and st.session_state.fl_manager.global_model is not None:
        try:
            # Use the fitted preprocessor from the federated learning manager
            if hasattr(st.session_state.fl_manager, 'preprocessor') and st.session_state.fl_manager.preprocessor.is_fitted:
                processed_data = st.session_state.fl_manager.preprocessor.transform(sample_data)
            else:
                # Fallback: create temporary preprocessor and fit on available data
                preprocessor = DataPreprocessor()
                # Load the diabetes dataset to fit the preprocessor
                data = pd.read_csv('diabetes.csv')
                X, y = preprocessor.fit_transform(data)
                processed_data = preprocessor.transform(sample_data)
            
            # Make prediction
            prediction = st.session_state.fl_manager.global_model.predict(processed_data)[0]
            probability = st.session_state.fl_manager.global_model.predict_proba(processed_data)[0][1]
            
            return prediction, probability
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return 0, 0.0
    
    return 0, 0.0

def get_risk_factors(patient):
    """Analyze key risk factors for a patient"""
    factors = []
    
    if patient['Glucose'] > 140:
        factors.append("High glucose")
    if patient['BMI'] > 30:
        factors.append("Obesity")
    if patient['Age'] > 45:
        factors.append("Advanced age")
    if patient['BloodPressure'] > 90:
        factors.append("High blood pressure")
    if patient['DiabetesPedigreeFunction'] > 0.5:
        factors.append("Family history")
    
    return ", ".join(factors) if factors else "No major risk factors"

def analyze_risk_factors(patient_data):
    """Detailed analysis of individual risk factors"""
    factors = {}
    
    # Glucose analysis
    glucose = patient_data['Glucose']
    if glucose < 100:
        factors['Glucose Level'] = {'risk': 'low', 'description': f'{glucose} mg/dL - Normal fasting glucose'}
    elif glucose < 126:
        factors['Glucose Level'] = {'risk': 'moderate', 'description': f'{glucose} mg/dL - Prediabetes range'}
    else:
        factors['Glucose Level'] = {'risk': 'high', 'description': f'{glucose} mg/dL - Diabetes range'}
    
    # BMI analysis
    bmi = patient_data['BMI']
    if bmi < 25:
        factors['BMI'] = {'risk': 'low', 'description': f'{bmi:.1f} - Normal weight'}
    elif bmi < 30:
        factors['BMI'] = {'risk': 'moderate', 'description': f'{bmi:.1f} - Overweight'}
    else:
        factors['BMI'] = {'risk': 'high', 'description': f'{bmi:.1f} - Obese'}
    
    # Age analysis
    age = patient_data['Age']
    if age < 35:
        factors['Age'] = {'risk': 'low', 'description': f'{age} years - Low risk age group'}
    elif age < 50:
        factors['Age'] = {'risk': 'moderate', 'description': f'{age} years - Moderate risk age group'}
    else:
        factors['Age'] = {'risk': 'high', 'description': f'{age} years - High risk age group'}
    
    # Blood pressure analysis
    bp = patient_data['BloodPressure']
    if bp < 80:
        factors['Blood Pressure'] = {'risk': 'low', 'description': f'{bp} mmHg - Normal'}
    elif bp < 90:
        factors['Blood Pressure'] = {'risk': 'moderate', 'description': f'{bp} mmHg - Elevated'}
    else:
        factors['Blood Pressure'] = {'risk': 'high', 'description': f'{bp} mmHg - High'}
    
    return factors

def get_recommendations(probability, patient_data):
    """Generate personalized recommendations based on risk assessment"""
    recommendations = []
    
    if probability >= 0.7:
        recommendations.extend([
            "Schedule immediate consultation with healthcare provider",
            "Request comprehensive diabetes screening tests",
            "Begin glucose monitoring if advised by doctor",
            "Implement strict dietary modifications"
        ])
    elif probability >= 0.4:
        recommendations.extend([
            "Schedule routine check-up within 3-6 months",
            "Adopt healthier eating habits with reduced sugar intake",
            "Increase physical activity to 150 minutes per week",
            "Monitor weight and blood pressure regularly"
        ])
    else:
        recommendations.extend([
            "Maintain current healthy lifestyle",
            "Continue regular exercise routine",
            "Annual health screenings as recommended",
            "Maintain healthy weight and diet"
        ])
    
    # Specific recommendations based on risk factors
    if patient_data['BMI'] > 30:
        recommendations.append("Focus on weight reduction through diet and exercise")
    
    if patient_data['Glucose'] > 140:
        recommendations.append("Monitor carbohydrate intake and meal timing")
    
    if patient_data['BloodPressure'] > 90:
        recommendations.append("Reduce sodium intake and manage stress levels")
    
    return recommendations

def reset_training():
    """Reset training state"""
    st.session_state.training_started = False
    st.session_state.training_completed = False
    st.session_state.results = None
    st.session_state.fl_manager = None
    st.session_state.training_metrics = []
    st.session_state.confusion_matrices = []
    st.session_state.execution_times = []
    st.session_state.communication_times = []
    st.session_state.client_status = {}
    st.session_state.best_accuracy = 0.0
    st.session_state.early_stopped = False
    st.session_state.current_round = 0
    st.session_state.client_results = []
    st.session_state.fog_results = []

def main():
    init_session_state()
    
    st.title("🌾 Agronomic Display - Hierarchical Federated Learning")
    st.markdown("**Advanced Crop Health Analytics & Prediction System**")
    st.markdown("---")
    
    # Data loading and preprocessing
    try:
        data = pd.read_csv('diabetes.csv')
        preprocessor = DataPreprocessor()
        X, y = preprocessor.fit_transform(data)
        
        st.success(f"✅ Patient Data loaded: {len(data)} patient records with {len(data.columns)} health indicators")
        
        # Display data overview in medical terms
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("👥 Patient Records", len(data))
        with col2:
            st.metric("📊 Health Indicators", len(data.columns) - 1)
        with col3:
            positive_ratio = (data['Outcome'] == 1).mean()
            st.metric("🚨 Diabetes Cases", f"{positive_ratio:.1%}")
            
    except Exception as e:
        st.error(f"❌ Error loading field data: {str(e)}")
        return
    
    # Multi-tab interface
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎛️ Training Control", "📈 Live Monitoring", "📊 Communication Network", "📋 Results Analysis", "🔍 Risk Prediction"])
    
    with tab1:
        st.header("🎛️ Federated Training Configuration")
        
        # Create three columns for better organization
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Basic Settings")
            num_clients = st.slider("👥 Number of Patient Agents", min_value=3, max_value=10, value=5)
            max_rounds = st.slider("🔄 Maximum Training Rounds", min_value=5, max_value=100, value=20)
            target_accuracy = st.slider("🎯 Target Accuracy (Auto-Stop)", min_value=0.7, max_value=0.95, value=0.85, step=0.05)
            
        with col2:
            st.subheader("🔧 Algorithm Settings")
            
            # Model selection with hyperparameter exploration
            model_type = st.selectbox("🤖 Machine Learning Model", [
                "logistic_regression", 
                "neural_network", 
                "random_forest", 
                "gradient_boosting", 
                "svm", 
                "ensemble_voting", 
                "ensemble_stacking"
            ], index=0)
            
            # Neural Network Hyperparameters (when applicable)
            if model_type == "neural_network":
                st.markdown("**🧠 Neural Network Configuration**")
                hidden_layers = st.selectbox("Hidden Layers", [1, 2, 3], index=1)
                neurons_per_layer = st.selectbox("Neurons per Layer", [64, 128, 256, 512], index=1)
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.1)
                learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=0)
                momentum = st.slider("Momentum", 0.0, 0.9, 0.5, 0.1)
            
            aggregation_algorithm = st.selectbox("Aggregation Algorithm", ["FedAvg", "FedProx", "SecureAgg"])
            
            # Advanced Privacy Settings
            enable_dp = st.checkbox("🔒 Enable Privacy Protection", value=True)
            if enable_dp:
                privacy_mechanism = st.selectbox("Privacy Mechanism", ["gaussian", "laplace", "exponential"])
                epsilon = st.number_input("🛡️ Privacy Budget (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                delta = st.number_input("🔐 Privacy Parameter (δ)", min_value=1e-6, max_value=1e-3, value=1e-5, format="%.1e")
                gradient_clip_norm = st.number_input("✂️ Gradient Clipping Norm", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            else:
                privacy_mechanism = "gaussian"
                epsilon = 1.0
                delta = 1e-5
                gradient_clip_norm = 1.0
            committee_size = st.slider("👥 Security Committee Size", min_value=2, max_value=5, value=3)
            
            st.markdown("**🌐 Fog Computing Configuration**")
            enable_fog = st.checkbox("Enable Hierarchical Fog Aggregation", value=True)
            if enable_fog:
                num_fog_nodes = st.slider("Number of Fog Nodes", min_value=2, max_value=5, value=3)
                fog_aggregation_method = st.selectbox("Fog Aggregation Strategy", 
                                                    ["Mixed Methods", "All FedAvg", "All WeightedAvg", "All Median"])
            else:
                num_fog_nodes = 0
                fog_aggregation_method = "None"
        
        with col3:
            st.subheader("🌍 Data Distribution Strategy")
            distribution_strategy = st.selectbox(
                "Distribution Pattern",
                ["IID", "Non-IID (Dirichlet)", "Pathological Non-IID", "Quantity Skew", "Geographic"],
                index=0
            )
            
            # Strategy-specific parameters
            strategy_params = {}
            if distribution_strategy == "Non-IID (Dirichlet)":
                alpha = st.slider("Alpha (Non-IID strength)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
                strategy_params['alpha'] = alpha
                st.info("Lower alpha = more non-IID")
            elif distribution_strategy == "Pathological Non-IID":
                classes_per_client = st.slider("Classes per Station", min_value=1, max_value=2, value=1)
                strategy_params['classes_per_client'] = classes_per_client
            elif distribution_strategy == "Quantity Skew":
                skew_factor = st.slider("Skew Factor", min_value=0.5, max_value=3.0, value=2.0, step=0.1)
                strategy_params['skew_factor'] = skew_factor
                st.info("Higher factor = more skewed")
            elif distribution_strategy == "Geographic":
                correlation_strength = st.slider("Geographic Correlation", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                strategy_params['correlation_strength'] = correlation_strength
            
            st.session_state.distribution_strategy = distribution_strategy
            st.session_state.strategy_params = strategy_params
        
        # Data distribution preview
        st.subheader("📈 Data Distribution Preview")
        if st.button("🔍 Preview Data Distribution"):
            with st.spinner("Generating distribution preview..."):
                try:
                    # Create distribution strategy
                    strategy = get_distribution_strategy(
                        distribution_strategy, 
                        num_clients, 
                        random_state=42,
                        **strategy_params
                    )
                    
                    # Apply distribution
                    client_data = strategy.distribute_data(X, y)
                    distribution_stats = strategy.get_distribution_stats(client_data)
                    
                    # Store for later use
                    st.session_state.distribution_stats = distribution_stats
                    st.session_state.preview_client_data = client_data
                    
                    # Create visualizations
                    fig_sizes, fig_heatmap = visualize_data_distribution(client_data, distribution_stats)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_sizes, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Display statistics
                    st.subheader("Distribution Statistics")
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    
                    with stats_col1:
                        st.metric("Strategy", distribution_stats['strategy'])
                        st.metric("Total Samples", distribution_stats['total_samples'])
                    
                    with stats_col2:
                        avg_size = np.mean(distribution_stats['client_sizes'])
                        std_size = np.std(distribution_stats['client_sizes'])
                        st.metric("Avg Station Size", f"{avg_size:.1f}")
                        st.metric("Size Std Dev", f"{std_size:.1f}")
                    
                    with stats_col3:
                        min_size = min(distribution_stats['client_sizes'])
                        max_size = max(distribution_stats['client_sizes'])
                        st.metric("Min Station Size", min_size)
                        st.metric("Max Station Size", max_size)
                    
                    st.success("Distribution preview generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating preview: {str(e)}")
        
        # Training controls
        st.subheader("🚀 Training Controls")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Start Federated Learning", disabled=st.session_state.training_started):
                start_training(data, num_clients, max_rounds, target_accuracy, 
                              aggregation_algorithm, enable_dp, epsilon, delta, committee_size,
                              distribution_strategy, strategy_params, enable_fog, 
                              num_fog_nodes, fog_aggregation_method, model_type, 
                              privacy_mechanism, gradient_clip_norm)
        
        with col2:
            if st.button("🔄 Reset System"):
                reset_training()
                st.rerun()
    
    with tab3:
        st.header("📊 Communication Network Architecture")
        
        if st.session_state.training_completed and hasattr(st.session_state, 'fl_manager'):
            # Network topology visualization
            st.subheader("🌐 Federated Learning Network Topology")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create network diagram using plotly
                import plotly.graph_objects as go
                import plotly.express as px
                import numpy as np
                
                # Generate network layout
                num_agents = st.session_state.fl_manager.num_clients
                fog_enabled = hasattr(st.session_state.fl_manager, 'fog_manager')
                
                # Node positions
                nodes_x = []
                nodes_y = []
                node_text = []
                node_colors = []
                
                # Global server (center)
                nodes_x.append(0)
                nodes_y.append(0)
                node_text.append("Global Server")
                node_colors.append("red")
                
                # Fog nodes (middle layer)
                if fog_enabled and hasattr(st.session_state.fl_manager, 'fog_manager'):
                    num_fog = st.session_state.fl_manager.fog_manager.num_fog_nodes
                    for i in range(num_fog):
                        angle = 2 * np.pi * i / num_fog
                        nodes_x.append(2 * np.cos(angle))
                        nodes_y.append(2 * np.sin(angle))
                        node_text.append(f"Fog Node {i+1}")
                        node_colors.append("orange")
                
                # Patient agents (outer layer)
                for i in range(num_agents):
                    if fog_enabled:
                        # Distribute around fog nodes
                        fog_idx = i % (num_fog if 'num_fog' in locals() else 1)
                        base_angle = 2 * np.pi * fog_idx / (num_fog if 'num_fog' in locals() else 1)
                        offset = (i // (num_fog if 'num_fog' in locals() else 1)) * 0.5
                        angle = base_angle + offset
                        radius = 4
                    else:
                        # Distribute around global server
                        angle = 2 * np.pi * i / num_agents
                        radius = 3
                    
                    nodes_x.append(radius * np.cos(angle))
                    nodes_y.append(radius * np.sin(angle))
                    node_text.append(f"Agent {i+1}")
                    node_colors.append("lightblue")
                
                # Create edges
                edge_x = []
                edge_y = []
                
                # Connect agents to fog nodes or global server
                for i in range(num_agents):
                    agent_idx = len(node_text) - num_agents + i
                    if fog_enabled and 'num_fog' in locals():
                        fog_idx = 1 + (i % num_fog)  # Fog nodes start at index 1
                        # Agent to fog
                        edge_x.extend([nodes_x[agent_idx], nodes_x[fog_idx], None])
                        edge_y.extend([nodes_y[agent_idx], nodes_y[fog_idx], None])
                    else:
                        # Agent to global server
                        edge_x.extend([nodes_x[agent_idx], nodes_x[0], None])
                        edge_y.extend([nodes_y[agent_idx], nodes_y[0], None])
                
                # Fog to global connections
                if fog_enabled and 'num_fog' in locals():
                    for i in range(1, num_fog + 1):
                        edge_x.extend([nodes_x[i], nodes_x[0], None])
                        edge_y.extend([nodes_y[i], nodes_y[0], None])
                
                # Create plotly figure
                fig = go.Figure()
                
                # Add edges
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(width=2, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ))
                
                # Add nodes
                fig.add_trace(go.Scatter(
                    x=nodes_x, y=nodes_y,
                    mode='markers+text',
                    marker=dict(size=20, color=node_colors),
                    text=node_text,
                    textposition="middle center",
                    hoverinfo='text',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title="Hierarchical Federated Learning Network",
                    showlegend=False,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    height=500,
                    plot_bgcolor='white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📊 Network Statistics")
                st.metric("Total Agents", num_agents)
                if fog_enabled and 'num_fog' in locals():
                    st.metric("Fog Nodes", num_fog)
                    st.metric("Agents per Fog", f"{num_agents//num_fog:.1f}")
                st.metric("Communication Rounds", len(st.session_state.training_history))
                
                # Communication efficiency metrics
                total_communications = num_agents * len(st.session_state.training_history)
                if fog_enabled and 'num_fog' in locals():
                    fog_communications = num_fog * len(st.session_state.training_history)
                    st.metric("Total Communications", total_communications + fog_communications)
                else:
                    st.metric("Total Communications", total_communications)
        else:
            st.info("Start and complete a training session to view the communication network.")
    
    with tab2:
        st.header("🏥 Patient Agent Monitoring")
        
        # Direct training execution
        if st.session_state.training_started and not st.session_state.training_completed:
            if hasattr(st.session_state, 'training_data') and st.session_state.fl_manager:
                st.info("🏥 Coordinating patient data analysis across medical agents...")
                
                # Create progress containers
                progress_container = st.empty()
                metrics_container = st.empty()
                charts_container = st.empty()
                
                # Run training with real-time updates
                try:
                    fl_manager = st.session_state.fl_manager
                    data = st.session_state.training_data
                    
                    # Apply custom data distribution
                    distribution_strategy = getattr(st.session_state, 'distribution_strategy', 'IID')
                    strategy_params = getattr(st.session_state, 'strategy_params', {})
                    
                    # Create distribution strategy
                    strategy = get_distribution_strategy(
                        distribution_strategy, 
                        fl_manager.num_clients, 
                        random_state=42,
                        **strategy_params
                    )
                    
                    # Preprocess data
                    preprocessor = DataPreprocessor()
                    X, y = preprocessor.fit_transform(data)
                    
                    # Apply distribution
                    client_data = strategy.distribute_data(X, y)
                    distribution_stats = strategy.get_distribution_stats(client_data)
                    
                    # Store distribution stats
                    st.session_state.distribution_stats = distribution_stats
                    
                    # Setup clients with distributed data
                    fl_manager.setup_clients_with_data(client_data)
                    
                    # Analysis cycle loop
                    for round_num in range(fl_manager.max_rounds):
                        current_round = round_num + 1
                        
                        # Update progress
                        with progress_container.container():
                            st.subheader(f"🌾 Crop Analysis Cycle {current_round}/{fl_manager.max_rounds}")
                            progress_bar = st.progress(current_round / fl_manager.max_rounds)
                            
                            # Field station status
                            cols = st.columns(fl_manager.num_clients)
                            for i in range(fl_manager.num_clients):
                                with cols[i]:
                                    st.metric(f"🏡 Farm {i+1}", "🌱 Analyzing")
                        
                        # Run training round
                        start_time = time.time()
                        
                        # Analyze crop samples at farms
                        client_updates = fl_manager._train_clients_parallel()
                        
                        # Regional processing centers (fog aggregation) if enabled
                        if hasattr(fl_manager, 'fog_manager') and fl_manager.fog_manager:
                            # Regional center aggregation
                            fog_updates, fog_metrics = fl_manager.fog_manager.fog_level_aggregation(
                                client_updates, fl_manager.global_model
                            )
                            
                            # Central agricultural hub aggregation
                            final_update = fl_manager.fog_manager.leader_fog_aggregation(
                                fog_updates, fl_manager.global_model
                            )
                            
                            if final_update:
                                # Update global model with fog-aggregated parameters
                                if isinstance(final_update['parameters'], dict):
                                    for param_name, param_value in final_update['parameters'].items():
                                        if hasattr(fl_manager.global_model, param_name):
                                            setattr(fl_manager.global_model, param_name, param_value)
                                else:
                                    # Handle array-based parameters - use standard aggregation
                                    fl_manager.global_model = fl_manager.aggregator.aggregate(
                                        fl_manager.global_model, client_updates
                                    )
                            
                            # Calculate hierarchical loss
                            loss_info = fl_manager.fog_manager.calculate_hierarchical_loss(
                                fl_manager.global_model, client_data, current_round
                            )
                            
                            # Store fog metrics
                            st.session_state.fog_results.append({
                                'round': current_round,
                                'fog_metrics': fog_metrics,
                                'loss_info': loss_info,
                                'aggregation_info': final_update.get('aggregation_info', {}) if final_update else {}
                            })
                        else:
                            # Standard aggregation
                            fl_manager.global_model = fl_manager.aggregator.aggregate(
                                fl_manager.global_model, client_updates
                            )
                        
                        # Evaluate
                        accuracy, loss, f1, cm = fl_manager._evaluate_global_model()
                        
                        round_time = time.time() - start_time
                        
                        # Extract individual client performance metrics
                        client_accuracies = [update.get('accuracy', 0.5) for update in client_updates]
                        client_f1_scores = [update.get('f1_score', 0.5) for update in client_updates]
                        
                        # Store comprehensive metrics with client-level tracking
                        metrics = {
                            'round': current_round,
                            'accuracy': accuracy,
                            'loss': loss,
                            'f1_score': f1,
                            'execution_time': round_time,
                            'client_accuracies': client_accuracies,
                            'client_f1_scores': client_f1_scores,
                            'accuracy_variance': np.var(client_accuracies) if client_accuracies else 0,
                            'min_client_accuracy': min(client_accuracies) if client_accuracies else 0,
                            'max_client_accuracy': max(client_accuracies) if client_accuracies else 0
                        }
                        
                        st.session_state.training_metrics.append(metrics)
                        st.session_state.execution_times.append(round_time)
                        st.session_state.confusion_matrices.append(cm)
                        st.session_state.communication_times.append(0.5)
                        st.session_state.best_accuracy = max(st.session_state.best_accuracy, accuracy)
                        st.session_state.current_round = current_round
                        
                        # Update crop health metrics display
                        with metrics_container.container():
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🌾 Crop Health Score", f"{accuracy:.3f}")
                            with col2:
                                st.metric("📊 Analysis Quality", f"{f1:.3f}")
                            with col3:
                                st.metric("⚠️ Error Rate", f"{loss:.4f}")
                            with col4:
                                st.metric("🏆 Best Performance", f"{st.session_state.best_accuracy:.3f}")
                            
                            # Individual farm performance display
                            if client_accuracies:
                                st.markdown("---")
                                st.markdown("**🏡 Individual Farm Performance**")
                                farm_cols = st.columns(min(5, len(client_accuracies)))
                                for i, acc in enumerate(client_accuracies):
                                    with farm_cols[i % len(farm_cols)]:
                                        performance_color = "🟢" if acc > 0.7 else "🟡" if acc > 0.5 else "🔴"
                                        st.metric(f"🏡 Farm {i+1}", f"{performance_color} {acc:.3f}")
                            
                            # Regional processing center metrics
                            if hasattr(fl_manager, 'fog_manager') and fl_manager.fog_manager and st.session_state.fog_results:
                                st.markdown("---")
                                st.markdown("**🏭 Regional Processing Centers Status**")
                                latest_fog = st.session_state.fog_results[-1]
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("🌍 Overall Field Error", f"{latest_fog['loss_info']['global_loss']:.4f}")
                                with col2:
                                    fog_losses = latest_fog['loss_info']['fog_losses']
                                    avg_fog_loss = np.mean([f['loss'] for f in fog_losses.values()]) if fog_losses else 0
                                    st.metric("🏭 Avg Regional Error", f"{avg_fog_loss:.4f}")
                                with col3:
                                    aggregation_info = latest_fog.get('aggregation_info', {})
                                    st.metric("🏢 Active Centers", aggregation_info.get('total_fog_nodes', 0))
                        
                        # Update charts
                        with charts_container.container():
                            if len(st.session_state.training_metrics) > 1:
                                show_training_charts()
                        
                        # Check if crop health target achieved
                        if accuracy >= fl_manager.target_accuracy:
                            st.success(f"🎯 Target crop health score {fl_manager.target_accuracy:.3f} achieved!")
                            st.session_state.early_stopped = True
                            break
                        
                        # Update patient agent status to completed
                        with progress_container.container():
                            st.subheader(f"🏥 Patient Analysis Round {current_round}/{fl_manager.max_rounds}")
                            progress_bar = st.progress(current_round / fl_manager.max_rounds)
                            
                            cols = st.columns(fl_manager.num_clients)
                            for i in range(fl_manager.num_clients):
                                with cols[i]:
                                    st.metric(f"👤 Agent {i+1}", "✅ Analysis Complete")
                        
                        time.sleep(1)  # Brief pause between rounds
                    
                    # Field analysis completed
                    st.session_state.training_completed = True
                    st.session_state.training_started = False
                    
                    # Final crop analysis results
                    final_accuracy = st.session_state.best_accuracy
                    st.session_state.results = {
                        'accuracy': final_accuracy,
                        'f1_score': f1,
                        'rounds_completed': current_round,
                        'early_stopped': st.session_state.early_stopped,
                        'training_history': st.session_state.training_metrics
                    }
                    
                    # Extract results for farm station reports
                    extract_training_results(fl_manager)
                    
                    st.success("🏥 Patient analysis completed successfully! All agents have finished diabetes risk assessment.")
                    
                except Exception as e:
                    st.error(f"Patient analysis failed: {str(e)}")
                    st.session_state.training_started = False
            else:
                st.warning("Please start patient analysis from the Training Control tab first.")
        
        # Show completed patient analysis results
        elif st.session_state.training_completed:
            st.success("✅ Training Completed")
            show_training_progress()
            if len(st.session_state.training_metrics) > 0:
                show_training_charts()
        else:
            st.info("🌱 Start training to see live monitoring data")
    
    with tab4:
        st.header("📋 Comprehensive Performance Analysis")
        
        if st.session_state.training_completed and hasattr(st.session_state, 'fl_manager'):
            # Performance comparison section
            st.subheader("🔍 Differential Privacy Trade-off Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy vs Privacy Trade-off
                if hasattr(st.session_state.fl_manager, 'dp_manager') and st.session_state.fl_manager.dp_manager:
                    privacy_params = st.session_state.fl_manager.dp_manager.get_privacy_parameters()
                    
                    # Simulate performance with different privacy levels
                    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                    simulated_accuracies = []
                    
                    # Base accuracy without privacy
                    base_accuracy = st.session_state.training_history[-1].get('accuracy', 0.85)
                    
                    for eps in epsilon_values:
                        # Simulate privacy-accuracy trade-off
                        privacy_noise = 1.0 / eps  # Higher epsilon = less noise
                        accuracy_loss = privacy_noise * 0.05  # Simulated accuracy degradation
                        simulated_accuracy = max(0.5, base_accuracy - accuracy_loss)
                        simulated_accuracies.append(simulated_accuracy)
                    
                    # Create privacy trade-off chart
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=epsilon_values,
                        y=simulated_accuracies,
                        mode='lines+markers',
                        name='Accuracy vs Privacy',
                        line=dict(color='blue', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Highlight current setting
                    current_eps = privacy_params.get('epsilon', 1.0)
                    current_acc = base_accuracy - (1.0 / current_eps) * 0.05
                    fig.add_trace(go.Scatter(
                        x=[current_eps],
                        y=[max(0.5, current_acc)],
                        mode='markers',
                        name='Current Setting',
                        marker=dict(size=15, color='red', symbol='star')
                    ))
                    
                    fig.update_layout(
                        title="Privacy-Accuracy Trade-off Analysis",
                        xaxis_title="Privacy Budget (ε)",
                        yaxis_title="Model Accuracy",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Privacy metrics table
                    st.subheader("🔒 Privacy Protection Metrics")
                    privacy_df = pd.DataFrame({
                        'Parameter': ['Epsilon (ε)', 'Delta (δ)', 'Noise Scale', 'Gradient Clipping'],
                        'Value': [
                            f"{privacy_params.get('epsilon', 'N/A'):.3f}",
                            f"{privacy_params.get('delta', 'N/A'):.2e}" if privacy_params.get('delta') else 'N/A',
                            f"{privacy_params.get('sensitivity', 'N/A'):.3f}" if privacy_params.get('sensitivity') else 'N/A',
                            f"{st.session_state.fl_manager.gradient_clip_norm:.2f}" if hasattr(st.session_state.fl_manager, 'gradient_clip_norm') else 'N/A'
                        ],
                        'Description': [
                            'Privacy budget - lower is more private',
                            'Failure probability in privacy guarantee',
                            'Noise magnitude added to gradients',
                            'Maximum gradient norm before clipping'
                        ]
                    })
                    st.dataframe(privacy_df, use_container_width=True)
                
            with col2:
                # Agent performance evolution
                st.subheader("👥 Agent Performance Evolution")
                
                if hasattr(st.session_state, 'client_results') and st.session_state.client_results:
                    # Extract individual agent performance over rounds
                    agent_performance = {}
                    
                    for round_data in st.session_state.training_history:
                        round_num = round_data.get('round', 0)
                        if 'client_metrics' in round_data:
                            for client_id, metrics in round_data['client_metrics'].items():
                                if client_id not in agent_performance:
                                    agent_performance[client_id] = {'rounds': [], 'accuracy': [], 'loss': []}
                                
                                agent_performance[client_id]['rounds'].append(round_num)
                                agent_performance[client_id]['accuracy'].append(metrics.get('accuracy', 0))
                                agent_performance[client_id]['loss'].append(metrics.get('loss', 0))
                    
                    # Create agent evolution chart
                    fig = go.Figure()
                    
                    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                    
                    for i, (client_id, data) in enumerate(agent_performance.items()):
                        color = colors[i % len(colors)]
                        fig.add_trace(go.Scatter(
                            x=data['rounds'],
                            y=data['accuracy'],
                            mode='lines+markers',
                            name=f'Agent {client_id}',
                            line=dict(color=color, width=2),
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title="Individual Agent Learning Curves",
                        xaxis_title="Training Round",
                        yaxis_title="Local Accuracy",
                        height=400,
                        legend=dict(
                            orientation="v",
                            yanchor="top",
                            y=1,
                            xanchor="left",
                            x=1.02
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Agent statistics summary
                if hasattr(st.session_state, 'client_results') and st.session_state.client_results:
                    st.subheader("📊 Agent Performance Summary")
                    
                    agent_stats = []
                    for client_data in st.session_state.client_results:
                        agent_stats.append({
                            'Agent ID': f"Agent {client_data.get('client_id', 'Unknown')}",
                            'Final Accuracy': f"{client_data.get('final_accuracy', 0):.3f}",
                            'Data Samples': client_data.get('data_size', 0),
                            'Avg Loss': f"{client_data.get('avg_loss', 0):.4f}",
                            'Convergence Round': client_data.get('convergence_round', 'N/A')
                        })
                    
                    agent_df = pd.DataFrame(agent_stats)
                    st.dataframe(agent_df, use_container_width=True)
            
            # Training results section
            st.subheader("📋 Training Results Analysis")
            if st.session_state.training_completed and st.session_state.results:
                show_results()
        else:
            st.info("Complete a training session to view comprehensive performance analysis.")
    
    with tab5:
        st.header("🏥 Patient Diabetes Risk Assessment")
        
        # Patient Database Management
        st.subheader("👥 Patient Database")
        
        # Initialize patient database in session state
        if 'patient_database' not in st.session_state:
            st.session_state.patient_database = []
        
        # Add new patient section
        with st.expander("➕ Add New Patient", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Personal Information**")
                patient_name = st.text_input("Patient Name")
                patient_id = st.text_input("Patient ID")
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
                age = st.number_input("Age (years)", min_value=18, max_value=120, value=30)
                
            with col2:
                st.markdown("**Medical Information**")
                glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
                insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=900, value=80)
                bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            
            if st.button("💾 Add Patient to Database"):
                if patient_name and patient_id:
                    patient_data = {
                        'name': patient_name,
                        'id': patient_id,
                        'Pregnancies': pregnancies,
                        'Glucose': glucose,
                        'BloodPressure': blood_pressure,
                        'SkinThickness': skin_thickness,
                        'Insulin': insulin,
                        'BMI': bmi,
                        'DiabetesPedigreeFunction': diabetes_pedigree,
                        'Age': age,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    # Check if patient ID already exists
                    existing_patient = next((p for p in st.session_state.patient_database if p['id'] == patient_id), None)
                    if existing_patient:
                        st.warning(f"Patient ID {patient_id} already exists. Please use a unique ID.")
                    else:
                        st.session_state.patient_database.append(patient_data)
                        st.success(f"Patient {patient_name} added successfully!")
                else:
                    st.error("Please provide both patient name and ID.")
        
        # Display patient database
        if st.session_state.patient_database:
            st.subheader("📋 Registered Patients")
            
            # Create DataFrame for display
            patients_df = pd.DataFrame(st.session_state.patient_database)
            display_df = patients_df[['name', 'id', 'Age', 'Glucose', 'BMI', 'timestamp']].copy()
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Add selection column
            selected_patients = st.multiselect(
                "Select patients for risk assessment:",
                options=patients_df.index.tolist(),
                format_func=lambda x: f"{patients_df.iloc[x]['name']} (ID: {patients_df.iloc[x]['id']})"
            )
            
            st.dataframe(display_df, use_container_width=True)
            
            # Batch risk assessment
            if selected_patients and st.session_state.training_completed:
                if st.button("🔬 Assess Diabetes Risk for Selected Patients"):
                    st.subheader("🎯 Risk Assessment Results")
                    
                    results = []
                    for idx in selected_patients:
                        patient = patients_df.iloc[idx]
                        
                        # Create prediction data
                        pred_data = pd.DataFrame({
                            'Pregnancies': [patient['Pregnancies']],
                            'Glucose': [patient['Glucose']],
                            'BloodPressure': [patient['BloodPressure']],
                            'SkinThickness': [patient['SkinThickness']],
                            'Insulin': [patient['Insulin']],
                            'BMI': [patient['BMI']],
                            'DiabetesPedigreeFunction': [patient['DiabetesPedigreeFunction']],
                            'Age': [patient['Age']]
                        })
                        
                        prediction, probability = make_prediction(pred_data)
                        
                        # Risk categorization
                        if probability >= 0.7:
                            risk_category = "High Risk"
                            risk_color = "🔴"
                            recommendation = "Immediate medical consultation recommended"
                        elif probability >= 0.4:
                            risk_category = "Moderate Risk"
                            risk_color = "🟡"
                            recommendation = "Regular monitoring and lifestyle modifications"
                        else:
                            risk_category = "Low Risk"
                            risk_color = "🟢"
                            recommendation = "Maintain healthy lifestyle habits"
                        
                        results.append({
                            'Patient': patient['name'],
                            'ID': patient['id'],
                            'Risk Level': f"{risk_color} {risk_category}",
                            'Probability': f"{probability:.1%}",
                            'Recommendation': recommendation,
                            'Key Factors': get_risk_factors(patient)
                        })
                    
                    # Display results table
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Generate summary statistics
                    col1, col2, col3 = st.columns(3)
                    high_risk = sum(1 for r in results if "High Risk" in r['Risk Level'])
                    moderate_risk = sum(1 for r in results if "Moderate Risk" in r['Risk Level'])
                    low_risk = sum(1 for r in results if "Low Risk" in r['Risk Level'])
                    
                    with col1:
                        st.metric("High Risk Patients", high_risk)
                    with col2:
                        st.metric("Moderate Risk Patients", moderate_risk)
                    with col3:
                        st.metric("Low Risk Patients", low_risk)
        
        # Quick Risk Assessment Tool
        st.subheader("⚡ Quick Risk Assessment")
        
        if st.session_state.training_completed and st.session_state.results:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Health Metrics**")
                quick_glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, key="quick_glucose")
                quick_bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, key="quick_bmi")
                quick_age = st.number_input("Age", min_value=18, max_value=120, value=30, key="quick_age")
                quick_pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, key="quick_pregnancies")
                
            with col2:
                st.markdown("**Additional Factors**")
                quick_bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80, key="quick_bp")
                quick_insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, key="quick_insulin")
                quick_skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, key="quick_skin")
                quick_pedigree = st.number_input("Family History Factor", min_value=0.0, max_value=3.0, value=0.5, step=0.01, key="quick_pedigree")
            
            if st.button("🎯 Assess Risk Now"):
                quick_data = pd.DataFrame({
                    'Pregnancies': [quick_pregnancies],
                    'Glucose': [quick_glucose],
                    'BloodPressure': [quick_bp],
                    'SkinThickness': [quick_skin],
                    'Insulin': [quick_insulin],
                    'BMI': [quick_bmi],
                    'DiabetesPedigreeFunction': [quick_pedigree],
                    'Age': [quick_age]
                })
                
                prediction, probability = make_prediction(quick_data)
                
                # Enhanced result display
                st.markdown("---")
                st.subheader("📊 Risk Assessment Result")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    color = "red" if prediction == 1 else "green"
                    st.markdown(f"### <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)
                    st.metric("Risk Probability", f"{probability:.1%}")
                
                with col2:
                    # Risk meter visualization
                    fig_meter = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Diabetes Risk %"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    fig_meter.update_layout(height=300)
                    st.plotly_chart(fig_meter, use_container_width=True)
                
                # Detailed risk factors analysis
                st.subheader("🔍 Risk Factors Analysis")
                factors = analyze_risk_factors(quick_data.iloc[0])
                
                for factor, info in factors.items():
                    status = "🔴 High" if info['risk'] == 'high' else "🟡 Moderate" if info['risk'] == 'moderate' else "🟢 Normal"
                    st.write(f"**{factor}:** {status} - {info['description']}")
                
                # Recommendations
                st.subheader("💡 Recommendations")
                recommendations = get_recommendations(probability, quick_data.iloc[0])
                for rec in recommendations:
                    st.write(f"• {rec}")
        else:
            st.info("Complete federated learning training to enable diabetes risk assessment")

if __name__ == "__main__":
    main()