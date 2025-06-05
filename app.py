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

def start_training(data, num_clients, max_rounds, target_accuracy, 
                  aggregation_algorithm, enable_dp, epsilon, delta, committee_size):
    """Start federated learning training with early stopping"""
    try:
        with st.spinner("Initializing Federated Learning System..."):
            # Create FL manager
            st.session_state.fl_manager = FederatedLearningManager(
                num_clients=num_clients,
                max_rounds=max_rounds,
                target_accuracy=target_accuracy,
                aggregation_algorithm=aggregation_algorithm,
                enable_dp=enable_dp,
                epsilon=epsilon,
                delta=delta,
                committee_size=committee_size
            )
            
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
            
            # Start training in background thread
            threading.Thread(
                target=run_training_loop,
                args=(st.session_state.fl_manager, data),
                daemon=True
            ).start()
            
            st.success("Training initiated successfully!")
            time.sleep(1)
            st.rerun()
            
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
    """Display training progress charts"""
    if not st.session_state.training_metrics:
        return
    
    st.header("📈 Training Analytics")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(st.session_state.training_metrics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy and F1 over rounds
        fig = go.Figure()
        if 'accuracy' in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(metrics_df) + 1)),
                y=metrics_df['accuracy'],
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=3)
            ))
        if 'f1_score' in metrics_df.columns:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(metrics_df) + 1)),
                y=metrics_df['f1_score'],
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='green', width=3)
            ))
        
        # Add target accuracy line
        if st.session_state.fl_manager:
            target = st.session_state.fl_manager.target_accuracy
            fig.add_hline(y=target, line_dash="dash", line_color="red", 
                         annotation_text=f"Target: {target:.3f}")
        
        fig.update_layout(
            title="Model Performance Over Rounds",
            xaxis_title="Training Round",
            yaxis_title="Score",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Execution times
        if st.session_state.execution_times:
            fig_time = px.bar(
                x=list(range(1, len(st.session_state.execution_times) + 1)),
                y=st.session_state.execution_times,
                title="Training Time per Round",
                labels={'x': 'Round', 'y': 'Time (seconds)'}
            )
            st.plotly_chart(fig_time, use_container_width=True)

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
        
        st.success(f"✅ Field Data loaded: {len(data)} crop samples with {len(data.columns)} health indicators")
        
        # Display data overview in agronomic terms
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌱 Field Samples", len(data))
        with col2:
            st.metric("📊 Health Indicators", len(data.columns) - 1)
        with col3:
            positive_ratio = (data['Outcome'] == 1).mean()
            st.metric("🚨 Risk Cases", f"{positive_ratio:.1%}")
            
    except Exception as e:
        st.error(f"❌ Error loading field data: {str(e)}")
        return
    
    # Multi-tab interface
    tab1, tab2, tab3, tab4 = st.tabs(["🎛️ Training Control", "📈 Live Monitoring", "📋 Results Analysis", "🔍 Risk Prediction"])
    
    with tab1:
        st.header("🎛️ Federated Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_clients = st.slider("🏢 Number of Field Stations", min_value=3, max_value=10, value=5)
            max_rounds = st.slider("🔄 Maximum Training Cycles", min_value=5, max_value=50, value=20)
            target_accuracy = st.slider("🎯 Target Accuracy (Auto-Stop)", min_value=0.7, max_value=0.95, value=0.85, step=0.05)
            
        with col2:
            aggregation_algorithm = st.selectbox("🔧 Aggregation Algorithm", ["FedAvg", "FedProx", "SecureAgg"])
            enable_dp = st.checkbox("🔒 Enable Privacy Protection", value=True)
            if enable_dp:
                epsilon = st.number_input("🛡️ Privacy Budget (ε)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                delta = st.number_input("🔐 Privacy Parameter (δ)", min_value=1e-6, max_value=1e-3, value=1e-5, format="%.1e")
            else:
                epsilon = delta = None
            committee_size = st.slider("👥 Security Committee Size", min_value=2, max_value=5, value=3)
        
        # Training controls
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Start Federated Learning", disabled=st.session_state.training_started):
                start_training(data, num_clients, max_rounds, target_accuracy, 
                              aggregation_algorithm, enable_dp, epsilon, delta, committee_size)
        
        with col2:
            if st.button("🔄 Reset System"):
                reset_training()
                st.rerun()
    
    with tab2:
        st.header("📈 Live Training Monitoring")
        
        # Auto-refresh for live monitoring
        if st.session_state.training_started and not st.session_state.training_completed:
            st.info("🔄 Training in progress - Auto-refreshing every 3 seconds")
            time.sleep(3)
            st.rerun()
        
        # Training Progress
        if st.session_state.training_started or st.session_state.training_completed:
            show_training_progress()
            
            # Show real-time charts
            if hasattr(st.session_state, 'training_metrics') and len(st.session_state.training_metrics) > 0:
                show_training_charts()
        else:
            st.info("🌱 Start training to see live monitoring data")
    
    with tab3:
        st.header("📋 Training Results Analysis")
        
        # Results
        if st.session_state.training_completed and st.session_state.results:
            show_results()
        else:
            st.info("🌾 Complete training to see detailed results analysis")
    
    with tab4:
        st.header("🔍 Crop Health Risk Prediction")
        
        if st.session_state.training_completed and st.session_state.results:
            # Prediction examples section
            st.subheader("📝 Example Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🟢 Low Risk Example")
                if st.button("Test Low Risk Sample"):
                    # Low risk example data
                    low_risk_data = pd.DataFrame({
                        'Pregnancies': [1],
                        'Glucose': [85],
                        'BloodPressure': [66],
                        'SkinThickness': [29],
                        'Insulin': [0],
                        'BMI': [26.6],
                        'DiabetesPedigreeFunction': [0.351],
                        'Age': [31]
                    })
                    
                    prediction, probability = make_prediction(low_risk_data)
                    
                    st.success(f"**Prediction:** {'High Risk' if prediction == 1 else 'Low Risk'}")
                    st.metric("Risk Probability", f"{probability:.1%}")
                    st.json(low_risk_data.iloc[0].to_dict())
            
            with col2:
                st.markdown("### 🔴 High Risk Example")
                if st.button("Test High Risk Sample"):
                    # High risk example data
                    high_risk_data = pd.DataFrame({
                        'Pregnancies': [6],
                        'Glucose': [148],
                        'BloodPressure': [72],
                        'SkinThickness': [35],
                        'Insulin': [0],
                        'BMI': [33.6],
                        'DiabetesPedigreeFunction': [0.627],
                        'Age': [50]
                    })
                    
                    prediction, probability = make_prediction(high_risk_data)
                    
                    if prediction == 1:
                        st.error(f"**Prediction:** High Risk")
                    else:
                        st.success(f"**Prediction:** Low Risk")
                    st.metric("Risk Probability", f"{probability:.1%}")
                    st.json(high_risk_data.iloc[0].to_dict())
            
            st.markdown("---")
            
            # Custom prediction interface
            st.subheader("🧪 Custom Field Sample Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Growth Conditions**")
                pregnancies = st.number_input("Growth Cycles", min_value=0, max_value=20, value=1)
                glucose = st.number_input("Nutrient Level", min_value=0, max_value=300, value=120)
                blood_pressure = st.number_input("Soil Pressure", min_value=0, max_value=200, value=80)
                skin_thickness = st.number_input("Leaf Thickness", min_value=0, max_value=100, value=20)
                
            with col2:
                st.markdown("**Environmental Factors**")
                insulin = st.number_input("Water Content", min_value=0, max_value=900, value=80)
                bmi = st.number_input("Plant Density Index", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
                diabetes_pedigree = st.number_input("Genetic Risk Factor", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
                age = st.number_input("Plant Maturity (days)", min_value=18, max_value=120, value=30)
            
            if st.button("🔬 Analyze Crop Sample"):
                # Create sample data
                sample_data = pd.DataFrame({
                    'Pregnancies': [pregnancies],
                    'Glucose': [glucose],
                    'BloodPressure': [blood_pressure],
                    'SkinThickness': [skin_thickness],
                    'Insulin': [insulin],
                    'BMI': [bmi],
                    'DiabetesPedigreeFunction': [diabetes_pedigree],
                    'Age': [age]
                })
                
                prediction, probability = make_prediction(sample_data)
                
                # Display prediction
                col1, col2 = st.columns(2)
                with col1:
                    risk_level = "High Risk" if prediction == 1 else "Low Risk"
                    color = "red" if prediction == 1 else "green"
                    st.markdown(f"### Crop Status: <span style='color: {color}'>{risk_level}</span>", unsafe_allow_html=True)
                    
                with col2:
                    st.metric("Risk Probability", f"{probability:.1%}")
                    
                # Risk interpretation
                if probability > 0.7:
                    st.warning("🚨 High disease risk detected. Recommend immediate field intervention.")
                elif probability > 0.3:
                    st.info("⚠️ Moderate risk. Consider preventive treatments and increased monitoring.")
                else:
                    st.success("✅ Healthy crop status. Continue current care protocols.")
        else:
            st.info("🌾 Complete training to enable crop risk prediction")

if __name__ == "__main__":
    main()