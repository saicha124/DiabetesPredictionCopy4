import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import json

from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from utils import calculate_metrics, plot_confusion_matrix

# Configure page
st.set_page_config(
    page_title="Hierarchical Federated Learning",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
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

def main():
    st.title("🏥 Hierarchical Federated Deep Learning")
    st.subheader("Diabetes Prediction with Differential Privacy & Committee-Based Security")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset section
    st.sidebar.subheader("Dataset Configuration")
    uploaded_file = st.sidebar.file_uploader(
        "Upload diabetes.csv", 
        type=['csv'],
        help="Upload the diabetes dataset for federated learning"
    )
    
    # Load default dataset if no file uploaded
    if uploaded_file is None:
        try:
            data = pd.read_csv('diabetes.csv')
            st.sidebar.success(f"Default dataset loaded: {len(data)} samples")
        except FileNotFoundError:
            st.sidebar.error("Default diabetes.csv not found. Please upload the dataset.")
            st.error("Please upload the diabetes.csv dataset to proceed.")
            return
    else:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Dataset uploaded: {len(data)} samples")
    
    # Federated Learning Configuration
    st.sidebar.subheader("Federated Learning Parameters")
    num_clients = st.sidebar.slider("Number of Clients", 2, 10, 5)
    max_rounds = st.sidebar.slider("Maximum Rounds", 5, 50, 20)
    target_accuracy = st.sidebar.slider("Target Accuracy", 0.7, 0.95, 0.85, 0.01)
    aggregation_algorithm = st.sidebar.selectbox(
        "Aggregation Algorithm", 
        ["FedAvg", "FedProx"]
    )
    
    # Differential Privacy Configuration
    st.sidebar.subheader("Differential Privacy")
    enable_dp = st.sidebar.checkbox("Enable Differential Privacy", value=True)
    epsilon = st.sidebar.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1) if enable_dp else None
    delta = st.sidebar.slider("Delta (δ)", 1e-6, 1e-3, 1e-5, format="%.2e") if enable_dp else None
    
    # Committee Security
    st.sidebar.subheader("Committee-Based Security")
    committee_size = st.sidebar.slider("Committee Size", 3, num_clients, min(5, num_clients))
    
    # Debug information (can be removed later)
    with st.expander("Debug Information", expanded=False):
        st.write("Training Started:", st.session_state.training_started)
        st.write("Training Completed:", st.session_state.training_completed)
        st.write("Results Available:", st.session_state.results is not None)
        st.write("Training Metrics Count:", len(st.session_state.training_metrics))
        st.write("Execution Times Count:", len(st.session_state.execution_times))
        st.write("Communication Times Count:", len(st.session_state.communication_times))
        if st.session_state.fl_manager:
            st.write("Current Round:", getattr(st.session_state.fl_manager, 'current_round', 0))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dataset Overview
        st.subheader("📊 Dataset Overview")
        if data is not None:
            st.write(f"**Dataset Shape:** {data.shape}")
            st.write(f"**Features:** {', '.join(data.columns[:-1])}")
            st.write(f"**Target Distribution:**")
            target_dist = data['Outcome'].value_counts()
            fig_dist = px.pie(
                values=target_dist.values, 
                names=['No Diabetes', 'Diabetes'],
                title="Target Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Training Section
        st.subheader("🚀 Federated Training")
        
        if not st.session_state.training_started:
            if st.button("Start Federated Training", type="primary"):
                start_training(
                    data, num_clients, max_rounds, target_accuracy,
                    aggregation_algorithm, enable_dp, epsilon, delta, committee_size
                )
        else:
            if st.session_state.training_completed:
                st.success("Training completed successfully!")
                if st.button("Reset Training"):
                    reset_training()
            else:
                st.info("Training in progress...")
                if st.button("Stop Training"):
                    st.session_state.training_started = False
                    reset_training()
        
        # Progress monitoring
        if st.session_state.training_started and not st.session_state.training_completed:
            show_training_progress()
        elif st.session_state.training_started and st.session_state.training_completed:
            st.success("Training completed successfully!")
            st.session_state.training_started = False  # Reset training flag
        
        # Results section
        if st.session_state.training_completed:
            st.success("🎉 Training Completed Successfully!")
            
            if st.session_state.results:
                show_results()
            else:
                st.warning("Training completed but results not available. Please check the logs.")
                
            # Show final metrics even if full results not available
            if st.session_state.training_metrics:
                st.subheader("Final Training Metrics")
                final_metrics = st.session_state.training_metrics[-1]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Final Accuracy", f"{final_metrics.get('accuracy', 0):.3f}")
                with col2:
                    st.metric("Final Loss", f"{final_metrics.get('loss', 0):.3f}")
                with col3:
                    st.metric("Final F1 Score", f"{final_metrics.get('f1_score', 0):.3f}")
                with col4:
                    st.metric("Total Rounds", len(st.session_state.training_metrics))
    
    with col2:
        # Patient Prediction
        st.subheader("🔮 Patient Prediction")
        
        if st.session_state.training_completed and st.session_state.fl_manager:
            with st.form("prediction_form"):
                st.write("Enter patient information:")
                
                pregnancies = st.number_input("Pregnancies", 0, 20, 0)
                glucose = st.number_input("Glucose", 0, 300, 120)
                blood_pressure = st.number_input("Blood Pressure", 0, 200, 80)
                skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
                insulin = st.number_input("Insulin", 0, 1000, 80)
                bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
                dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
                age = st.number_input("Age", 0, 120, 30)
                
                predict_btn = st.form_submit_button("Predict Diabetes Risk", type="primary")
                
                if predict_btn:
                    patient_data = np.array([[pregnancies, glucose, blood_pressure, 
                                           skin_thickness, insulin, bmi, dpf, age]])
                    
                    prediction, probability = make_prediction(patient_data)
                    
                    st.subheader("🏥 Prediction Results")
                    if prediction == 1:
                        st.error(f"**High Risk of Diabetes**")
                        st.write(f"Probability: {probability:.2%}")
                    else:
                        st.success(f"**Low Risk of Diabetes**")
                        st.write(f"Probability of No Diabetes: {1-probability:.2%}")
                    
                    # Risk gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Diabetes Risk %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 25], 'color': "lightgreen"},
                                {'range': [25, 50], 'color': "yellow"},
                                {'range': [50, 75], 'color': "orange"},
                                {'range': [75, 100], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 75}}))
                    st.plotly_chart(fig_gauge, use_container_width=True)
        else:
            st.info("Complete training first to enable predictions")

def start_training(data, num_clients, max_rounds, target_accuracy, 
                  aggregation_algorithm, enable_dp, epsilon, delta, committee_size):
    """Start federated learning training"""
    try:
        st.session_state.training_started = True
        st.session_state.training_completed = False
        st.session_state.training_metrics = []
        st.session_state.confusion_matrices = []
        st.session_state.execution_times = []
        st.session_state.communication_times = []
        st.session_state.client_status = {}
        st.session_state.results = None
        
        # Initialize federated learning manager
        fl_manager = FederatedLearningManager(
            num_clients=num_clients,
            max_rounds=max_rounds,
            target_accuracy=target_accuracy,
            aggregation_algorithm=aggregation_algorithm,
            enable_dp=enable_dp,
            epsilon=epsilon,
            delta=delta,
            committee_size=committee_size
        )
        
        st.session_state.fl_manager = fl_manager
        
        # Run training directly without threading to avoid session state issues
        with st.spinner("Training federated learning model..."):
            results = fl_manager.train(data)
            st.session_state.results = results
            st.session_state.training_completed = True
            st.session_state.training_started = False
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")
        st.session_state.training_started = False
        st.session_state.training_completed = False

def run_training_loop(fl_manager, data):
    """Run the training loop in background"""
    try:
        results = fl_manager.train(data)
        
        # Force update session state
        st.session_state.results = results
        st.session_state.training_completed = True
        st.session_state.training_started = False
        
        print(f"Training completed successfully! Final accuracy: {results.get('final_accuracy', 0):.3f}")
        
    except Exception as e:
        st.session_state.training_started = False
        st.session_state.training_completed = False
        print(f"Training failed: {str(e)}")

def show_training_progress():
    """Display real-time training progress"""
    if st.session_state.fl_manager:
        st.subheader("🚀 Training Progress")
        
        # Overall progress section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Round progress
            current_round = getattr(st.session_state.fl_manager, 'current_round', 0)
            max_rounds = st.session_state.fl_manager.max_rounds
            round_progress = st.progress(min(current_round / max_rounds, 1.0))
            st.write(f"Round {current_round} of {max_rounds}")
            
        with col2:
            # Target accuracy indicator
            target_accuracy = st.session_state.fl_manager.target_accuracy
            current_accuracy = 0.0
            if st.session_state.training_metrics:
                current_accuracy = st.session_state.training_metrics[-1].get('accuracy', 0)
            
            accuracy_progress = st.progress(min(current_accuracy / target_accuracy, 1.0))
            st.write(f"Accuracy: {current_accuracy:.3f} / {target_accuracy:.3f}")
        
        # Client progress section
        st.subheader("📊 Client Training Status")
        num_clients = st.session_state.fl_manager.num_clients
        
        # Create client progress bars
        client_cols = st.columns(min(num_clients, 5))  # Max 5 columns for display
        
        for i in range(num_clients):
            col_idx = i % len(client_cols)
            with client_cols[col_idx]:
                # Get client status
                client_status = st.session_state.client_status.get(i, 'waiting')
                
                # Calculate progress based on status
                if client_status == 'waiting':
                    progress_val = 0.0
                    status_color = "🔄"
                elif client_status == 'training':
                    progress_val = 0.5
                    status_color = "🟡"
                elif client_status == 'completed':
                    progress_val = 1.0
                    status_color = "🟢"
                else:  # failed
                    progress_val = 0.0
                    status_color = "🔴"
                
                st.progress(progress_val)
                st.caption(f"{status_color} Client {i}")
                st.caption(f"Status: {client_status.title()}")
        
        # Current metrics display
        if st.session_state.training_metrics:
            latest_metrics = st.session_state.training_metrics[-1]
            
            st.subheader("📈 Current Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Round", current_round, delta=1 if current_round > 1 else None)
            with col2:
                accuracy = latest_metrics.get('accuracy', 0)
                prev_accuracy = 0
                if len(st.session_state.training_metrics) > 1:
                    prev_accuracy = st.session_state.training_metrics[-2].get('accuracy', 0)
                delta_accuracy = accuracy - prev_accuracy if len(st.session_state.training_metrics) > 1 else None
                st.metric("Accuracy", f"{accuracy:.3f}", delta=f"{delta_accuracy:.3f}" if delta_accuracy else None)
            with col3:
                loss = latest_metrics.get('loss', 0)
                prev_loss = 0
                if len(st.session_state.training_metrics) > 1:
                    prev_loss = st.session_state.training_metrics[-2].get('loss', 0)
                delta_loss = loss - prev_loss if len(st.session_state.training_metrics) > 1 else None
                st.metric("Loss", f"{loss:.3f}", delta=f"{delta_loss:.3f}" if delta_loss else None)
            with col4:
                f1 = latest_metrics.get('f1_score', 0)
                prev_f1 = 0
                if len(st.session_state.training_metrics) > 1:
                    prev_f1 = st.session_state.training_metrics[-2].get('f1_score', 0)
                delta_f1 = f1 - prev_f1 if len(st.session_state.training_metrics) > 1 else None
                st.metric("F1 Score", f"{f1:.3f}", delta=f"{delta_f1:.3f}" if delta_f1 else None)
        
        # Real-time charts
        if len(st.session_state.training_metrics) > 1:
            st.subheader("📊 Real-time Training Charts")
            show_training_charts()
            
        # Execution and Communication Times
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.execution_times:
                st.subheader("⏱️ Fog Execution Times")
                current_exec_time = st.session_state.execution_times[-1] if st.session_state.execution_times else 0
                avg_exec_time = np.mean(st.session_state.execution_times) if st.session_state.execution_times else 0
                st.metric("Current Round Time", f"{current_exec_time:.2f}s")
                st.metric("Average Time", f"{avg_exec_time:.2f}s")
        
        with col2:
            if st.session_state.communication_times:
                st.subheader("📡 Communication Times")
                current_comm_time = st.session_state.communication_times[-1] if st.session_state.communication_times else 0
                avg_comm_time = np.mean(st.session_state.communication_times) if st.session_state.communication_times else 0
                st.metric("Current Comm Time", f"{current_comm_time:.2f}s")
                st.metric("Average Comm Time", f"{avg_comm_time:.2f}s")

def show_training_charts():
    """Display training progress charts"""
    if not st.session_state.training_metrics:
        return
    
    df_metrics = pd.DataFrame(st.session_state.training_metrics)
    
    # Main training metrics chart
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy and Loss chart
        fig_metrics = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Training Accuracy', 'Training Loss'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Accuracy
        fig_metrics.add_trace(
            go.Scatter(x=df_metrics['round'], y=df_metrics['accuracy'], 
                      mode='lines+markers', name='Accuracy', 
                      line=dict(color='blue', width=3)),
            row=1, col=1
        )
        
        # Add target accuracy line
        target_acc = st.session_state.fl_manager.target_accuracy if st.session_state.fl_manager else 0.85
        fig_metrics.add_hline(y=target_acc, line_dash="dash", line_color="green", 
                             annotation_text="Target Accuracy")
        
        # Loss
        fig_metrics.add_trace(
            go.Scatter(x=df_metrics['round'], y=df_metrics['loss'], 
                      mode='lines+markers', name='Loss', 
                      line=dict(color='red', width=3)),
            row=2, col=1
        )
        
        fig_metrics.update_layout(height=400, showlegend=False, 
                                 title_text="Training Performance")
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    with col2:
        # F1 Score and Communication Time
        fig_other = make_subplots(
            rows=2, cols=1,
            subplot_titles=('F1 Score', 'Communication Time'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # F1 Score
        fig_other.add_trace(
            go.Scatter(x=df_metrics['round'], y=df_metrics['f1_score'], 
                      mode='lines+markers', name='F1 Score', 
                      line=dict(color='green', width=3)),
            row=1, col=1
        )
        
        # Communication Time
        if st.session_state.communication_times:
            comm_rounds = list(range(1, len(st.session_state.communication_times) + 1))
            fig_other.add_trace(
                go.Scatter(x=comm_rounds, y=st.session_state.communication_times, 
                          mode='lines+markers', name='Comm Time', 
                          line=dict(color='purple', width=3)),
                row=2, col=1
            )
        
        fig_other.update_layout(height=400, showlegend=False, 
                               title_text="Additional Metrics")
        st.plotly_chart(fig_other, use_container_width=True)
    
    # Execution time chart
    if st.session_state.execution_times:
        st.subheader("Fog Execution Times Per Round")
        exec_rounds = list(range(1, len(st.session_state.execution_times) + 1))
        fig_exec = go.Figure()
        
        fig_exec.add_trace(go.Scatter(
            x=exec_rounds, 
            y=st.session_state.execution_times,
            mode='lines+markers',
            name='Execution Time',
            line=dict(color='orange', width=3),
            fill='tonexty'
        ))
        
        fig_exec.update_layout(
            title="Fog Execution Times",
            xaxis_title="Round",
            yaxis_title="Time (seconds)",
            height=300
        )
        
        st.plotly_chart(fig_exec, use_container_width=True)

def show_results():
    """Display final training results"""
    results = st.session_state.results
    
    st.subheader("📈 Training Results")
    
    # Final metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Final Accuracy", f"{results['final_accuracy']:.3f}")
    with col2:
        st.metric("Final Loss", f"{results['final_loss']:.3f}")
    with col3:
        st.metric("Rounds Completed", results['rounds_completed'])
    with col4:
        st.metric("Total Time", f"{results['total_time']:.2f}s")
    
    # Performance charts
    if st.session_state.training_metrics:
        show_training_charts()
    
    # Confusion Matrix
    if st.session_state.confusion_matrices:
        st.subheader("🎯 Final Confusion Matrix")
        final_cm = st.session_state.confusion_matrices[-1]
        fig_cm = plot_confusion_matrix(final_cm)
        st.pyplot(fig_cm)
    
    # Execution and Communication Times
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.execution_times:
            st.subheader("⏱️ Fog Execution Times")
            fig_exec = px.line(
                x=list(range(len(st.session_state.execution_times))),
                y=st.session_state.execution_times,
                title="Execution Time per Round",
                labels={'x': 'Round', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig_exec, use_container_width=True)
    
    with col2:
        if st.session_state.communication_times:
            st.subheader("📡 Communication Times")
            fig_comm = px.line(
                x=list(range(len(st.session_state.communication_times))),
                y=st.session_state.communication_times,
                title="Communication Time per Round",
                labels={'x': 'Round', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig_comm, use_container_width=True)

def make_prediction(patient_data):
    """Make prediction for new patient data"""
    if st.session_state.fl_manager and hasattr(st.session_state.fl_manager, 'global_model') and st.session_state.fl_manager.global_model is not None:
        try:
            # Use the fitted preprocessor from the federated learning manager
            if hasattr(st.session_state.fl_manager, 'preprocessor') and st.session_state.fl_manager.preprocessor.is_fitted:
                processed_data = st.session_state.fl_manager.preprocessor.transform(patient_data)
            else:
                # Fallback: create temporary preprocessor and fit on available data
                preprocessor = DataPreprocessor()
                # Load the diabetes dataset to fit the preprocessor
                data = pd.read_csv('diabetes.csv')
                X, y = preprocessor.fit_transform(data)
                processed_data = preprocessor.transform(patient_data)
            
            # Make prediction
            prediction = st.session_state.fl_manager.global_model.predict(processed_data)[0]
            probability = st.session_state.fl_manager.global_model.predict_proba(processed_data)[0][1]
            
            return prediction, probability
        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return 0, 0.0
    
    return 0, 0.0

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
    st.rerun()

if __name__ == "__main__":
    main()
