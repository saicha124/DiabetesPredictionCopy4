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
        
        # Results section
        if st.session_state.training_completed and st.session_state.results:
            show_results()
    
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
        
        # Run training in a separate thread
        training_thread = threading.Thread(
            target=run_training_loop,
            args=(fl_manager, data)
        )
        training_thread.daemon = True
        training_thread.start()
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")
        st.session_state.training_started = False

def run_training_loop(fl_manager, data):
    """Run the training loop in background"""
    try:
        results = fl_manager.train(data)
        st.session_state.results = results
        st.session_state.training_completed = True
        st.session_state.training_started = False
    except Exception as e:
        st.session_state.training_started = False
        st.error(f"Training failed: {str(e)}")

def show_training_progress():
    """Display real-time training progress"""
    if st.session_state.fl_manager:
        # Progress bars
        round_progress = st.progress(0)
        client_progress = st.progress(0)
        
        # Metrics placeholders
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Update progress periodically
        while st.session_state.training_started and not st.session_state.training_completed:
            try:
                # Get current progress from FL manager
                current_round = getattr(st.session_state.fl_manager, 'current_round', 0)
                max_rounds = st.session_state.fl_manager.max_rounds
                
                # Update progress bars
                round_progress.progress(min(current_round / max_rounds, 1.0))
                
                # Display current metrics
                if st.session_state.training_metrics:
                    latest_metrics = st.session_state.training_metrics[-1]
                    
                    with metrics_placeholder.container():
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Round", current_round)
                        with col2:
                            st.metric("Accuracy", f"{latest_metrics.get('accuracy', 0):.3f}")
                        with col3:
                            st.metric("Loss", f"{latest_metrics.get('loss', 0):.3f}")
                        with col4:
                            st.metric("F1 Score", f"{latest_metrics.get('f1_score', 0):.3f}")
                
                # Update training chart
                if len(st.session_state.training_metrics) > 1:
                    with chart_placeholder.container():
                        show_training_charts()
                
                time.sleep(2)  # Update every 2 seconds
                
            except Exception as e:
                st.error(f"Error updating progress: {str(e)}")
                break

def show_training_charts():
    """Display training progress charts"""
    if not st.session_state.training_metrics:
        return
    
    df_metrics = pd.DataFrame(st.session_state.training_metrics)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy', 'Loss', 'F1 Score', 'Execution Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=df_metrics.index, y=df_metrics['accuracy'], 
                  mode='lines+markers', name='Accuracy'),
        row=1, col=1
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=df_metrics.index, y=df_metrics['loss'], 
                  mode='lines+markers', name='Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # F1 Score
    fig.add_trace(
        go.Scatter(x=df_metrics.index, y=df_metrics['f1_score'], 
                  mode='lines+markers', name='F1 Score', line=dict(color='green')),
        row=2, col=1
    )
    
    # Execution Time
    if st.session_state.execution_times:
        fig.add_trace(
            go.Scatter(x=list(range(len(st.session_state.execution_times))), 
                      y=st.session_state.execution_times, 
                      mode='lines+markers', name='Execution Time', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

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
    if st.session_state.fl_manager and hasattr(st.session_state.fl_manager, 'global_model'):
        # Preprocess the data
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.transform(patient_data)
        
        # Make prediction
        prediction = st.session_state.fl_manager.global_model.predict(processed_data)[0]
        probability = st.session_state.fl_manager.global_model.predict_proba(processed_data)[0][1]
        
        return prediction, probability
    
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
