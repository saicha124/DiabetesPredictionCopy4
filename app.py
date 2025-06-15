import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
import os

# Import custom modules
from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from translations import get_translation

# Import visualization modules with error handling
try:
    from journey_visualization import JourneyVisualizer
except ImportError:
    JourneyVisualizer = None

try:
    from client_visualization import ClientPerformanceVisualizer
except ImportError:
    ClientPerformanceVisualizer = None

try:
    from advanced_client_analytics import AdvancedClientAnalytics
except ImportError:
    AdvancedClientAnalytics = None

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'language': 'en',
        'data_loaded': False,
        'training_started': False,
        'training_completed': False,
        'training_in_progress': False,
        'current_tab_index': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize visualizers with error handling
    if 'journey_visualizer' not in st.session_state:
        if JourneyVisualizer:
            st.session_state.journey_visualizer = JourneyVisualizer()
        else:
            st.session_state.journey_visualizer = None
    
    if 'client_visualizer' not in st.session_state:
        if ClientPerformanceVisualizer:
            st.session_state.client_visualizer = ClientPerformanceVisualizer()
        else:
            st.session_state.client_visualizer = None
    
    if 'advanced_analytics' not in st.session_state:
        if AdvancedClientAnalytics:
            st.session_state.advanced_analytics = AdvancedClientAnalytics()
        else:
            st.session_state.advanced_analytics = None

def main():
    st.set_page_config(
        page_title="Hierarchical Federated Learning Platform",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # Header with language selector
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        if st.session_state.language == 'fr':
            st.title("🏥 Plateforme d'Apprentissage Fédéré Hiérarchique")
            st.markdown("**Prédiction du diabète avec préservation de la vie privée pour les établissements médicaux**")
        else:
            st.title("🏥 Hierarchical Federated Learning Platform")
            st.markdown("**Privacy-Preserving Diabetes Prediction for Medical Facilities**")
    
    with col3:
        language_options = {'English': 'en', 'Français': 'fr'}
        selected_language = st.selectbox(
            "Language/Langue:",
            options=list(language_options.keys()),
            index=0 if st.session_state.language == 'en' else 1,
            key="language_selector"
        )
        
        if language_options[selected_language] != st.session_state.language:
            st.session_state.language = language_options[selected_language]
            st.rerun()
    
    # Load dataset
    if not st.session_state.data_loaded:
        try:
            st.session_state.data = pd.read_csv('diabetes.csv')
            st.session_state.data_loaded = True
            
            # Initialize preprocessing
            if 'preprocessor' not in st.session_state:
                st.session_state.preprocessor = DataPreprocessor()
                st.session_state.training_data = st.session_state.data
                
        except FileNotFoundError:
            st.error(get_translation("dataset_not_found", st.session_state.language))
            return
        except Exception as e:
            st.error(get_translation("failed_to_load_dataset", st.session_state.language, error=str(e)))
            return

    # Initialize current tab index
    if 'current_tab_index' not in st.session_state:
        st.session_state.current_tab_index = 0
    
    # Tab names array
    tab_names = [
        get_translation("tab_training", st.session_state.language),
        get_translation("tab_monitoring", st.session_state.language), 
        get_translation("tab_visualization", st.session_state.language),
        get_translation("tab_analytics", st.session_state.language),
        get_translation("tab_risk", st.session_state.language),
        get_translation("tab_facility", st.session_state.language),
        get_translation("tab_graph_viz", st.session_state.language),
        get_translation("tab_advanced_analytics", st.session_state.language)
    ]
    
    # Tab navigation with arrows
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col1:
        if st.button("◀ Précédent" if st.session_state.language == 'fr' else "◀ Previous", key="prev_tab"):
            if st.session_state.current_tab_index > 0:
                st.session_state.current_tab_index -= 1
                st.rerun()
    
    with col2:
        current_tab_name = tab_names[st.session_state.current_tab_index]
        if st.session_state.language == 'fr':
            st.markdown(f"**Onglet actuel:** {current_tab_name} ({st.session_state.current_tab_index + 1}/8)")
        else:
            st.markdown(f"**Current tab:** {current_tab_name} ({st.session_state.current_tab_index + 1}/8)")
    
    with col3:
        if st.button("Suivant ▶" if st.session_state.language == 'fr' else "Next ▶", key="next_tab"):
            if st.session_state.current_tab_index < len(tab_names) - 1:
                st.session_state.current_tab_index += 1
                st.rerun()

    # Display content based on current tab index
    if st.session_state.current_tab_index == 0:
        # Tab 1: Training
        st.header("🎛️ " + get_translation("tab_training", st.session_state.language))
        
        if st.session_state.data_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(get_translation("federated_learning_configuration", st.session_state.language))
                
                num_clients = st.slider(get_translation("number_medical_facilities", st.session_state.language), 
                                       min_value=3, max_value=10, value=5)
                
                max_rounds = st.slider(get_translation("training_rounds", st.session_state.language), 
                                     min_value=5, max_value=50, value=20)
                
                target_accuracy = st.slider(get_translation("target_accuracy", st.session_state.language), 
                                           min_value=0.7, max_value=0.99, value=0.85, step=0.01)
            
            with col2:
                st.subheader(get_translation("advanced_settings", st.session_state.language))
                
                aggregation_algo = st.selectbox(
                    get_translation("aggregation_algorithm", st.session_state.language),
                    ["FedAvg", "FedProx", "Secure"]
                )
                
                model_type = st.selectbox(
                    get_translation("model_type", st.session_state.language),
                    ["logistic_regression", "random_forest", "neural_network"]
                )
            
            if not st.session_state.training_started:
                if st.button(get_translation("start_federated_training", st.session_state.language), 
                           type="primary", use_container_width=True):
                    
                    # Initialize federated learning
                    st.session_state.fl_manager = FederatedLearningManager(
                        num_clients=num_clients,
                        aggregation_algorithm=aggregation_algo,
                        model_type=model_type
                    )
                    
                    st.session_state.training_started = True
                    st.session_state.training_in_progress = True
                    st.session_state.max_rounds = max_rounds
                    st.session_state.target_accuracy = target_accuracy
                    st.session_state.current_training_round = 0
                    st.session_state.training_metrics = []
                    st.session_state.best_accuracy = 0
                    
                    st.rerun()
            
            # Training progress display
            if st.session_state.training_started and st.session_state.training_in_progress:
                st.subheader(get_translation("training_progress", st.session_state.language))
                
                # Simulate training progress
                current_round = st.session_state.current_training_round
                max_rounds = st.session_state.max_rounds
                
                progress = current_round / max_rounds
                st.progress(progress, text=f"Round {current_round}/{max_rounds}")
                
                if current_round < max_rounds:
                    # Simulate one training round
                    time.sleep(1)
                    st.session_state.current_training_round += 1
                    
                    # Simulate metrics
                    accuracy = 0.6 + (0.3 * progress) + np.random.normal(0, 0.02)
                    accuracy = max(0.6, min(0.95, accuracy))
                    
                    st.session_state.training_metrics.append({
                        'round': current_round,
                        'accuracy': accuracy,
                        'loss': 0.7 - (0.4 * progress) + np.random.normal(0, 0.05)
                    })
                    
                    if accuracy > st.session_state.best_accuracy:
                        st.session_state.best_accuracy = accuracy
                    
                    st.rerun()
                else:
                    st.session_state.training_completed = True
                    st.session_state.training_in_progress = False
                    st.success(get_translation("training_completed", st.session_state.language))

    elif st.session_state.current_tab_index == 1:
        # Tab 2: Monitoring
        st.header("🏥 " + get_translation("medical_station_monitoring", st.session_state.language))
        
        if st.session_state.training_completed:
            st.success(get_translation("training_completed_successfully", st.session_state.language))
            
            # Display final results
            if st.session_state.training_metrics:
                final_accuracy = st.session_state.training_metrics[-1]['accuracy']
                st.metric("Final Accuracy", f"{final_accuracy:.2%}")
        else:
            st.info(get_translation("start_training_first", st.session_state.language))

    elif st.session_state.current_tab_index == 2:
        # Tab 3: Visualization
        st.header("🗺️ " + get_translation("interactive_learning_journey_visualization", st.session_state.language))
        
        if st.session_state.training_completed:
            if st.session_state.journey_visualizer:
                st.session_state.journey_visualizer.create_interactive_journey()
            else:
                st.info("Journey visualization feature available after training completion.")
        else:
            st.info(get_translation("complete_training_see_visualization", st.session_state.language))

    elif st.session_state.current_tab_index == 3:
        # Tab 4: Analytics
        st.header("📊 " + get_translation("performance_analysis", st.session_state.language))
        
        if st.session_state.training_completed and st.session_state.training_metrics:
            # Create metrics visualization
            metrics_df = pd.DataFrame(st.session_state.training_metrics)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.line(metrics_df, x='round', y='accuracy', title='Training Accuracy')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.line(metrics_df, x='round', y='loss', title='Training Loss')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(get_translation("complete_training_see_performance", st.session_state.language))

    elif st.session_state.current_tab_index == 4:
        # Tab 5: Risk Prediction
        st.header(f"🩺 {get_translation('patient_risk_prediction_explainer', st.session_state.language)}")
        
        if st.session_state.training_completed:
            st.subheader(f"🔍 {get_translation('patient_risk_analysis', st.session_state.language)}")
            st.info(get_translation("enter_patient_data_for_risk_assessment", st.session_state.language))
        else:
            st.warning(get_translation("complete_federated_training", st.session_state.language))

    elif st.session_state.current_tab_index == 5:
        # Tab 6: Facility Analytics
        st.header("🏥 " + get_translation("tab_facility", st.session_state.language))
        
        if st.session_state.training_completed:
            st.subheader("📊 " + get_translation("feature_correlation_analysis", st.session_state.language))
            
            if st.session_state.data_loaded:
                # Create correlation matrix
                numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
                corr_matrix = st.session_state.data[numeric_cols].corr()
                
                fig = px.imshow(corr_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title=get_translation("correlation_matrix", st.session_state.language))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(get_translation("complete_training_see_analytics", st.session_state.language))

    elif st.session_state.current_tab_index == 6:
        # Tab 7: Graph Visualization
        if st.session_state.language == 'fr':
            st.header("🌐 Visualisation Graphique")
        else:
            st.header("🌐 Graph Visualization")
        
        # Visualization options
        if st.session_state.language == 'fr':
            viz_option = st.selectbox(
                "Type de visualisation:",
                ["Réseau de Fédération", "Topologie Hiérarchique", "Flux de Communication", "Architecture 3D"]
            )
        else:
            viz_option = st.selectbox(
                "Visualization Type:",
                ["Federation Network", "Hierarchical Topology", "Communication Flow", "3D Architecture"]
            )
        
        if viz_option in ["Federation Network", "Réseau de Fédération"]:
            if st.session_state.language == 'fr':
                st.subheader("🔗 Réseau de Fédération")
                st.info("Visualisation interactive du réseau d'apprentissage fédéré avec les connexions entre les établissements médicaux.")
            else:
                st.subheader("🔗 Federation Network") 
                st.info("Interactive visualization of the federated learning network with connections between medical facilities.")
        elif viz_option in ["Hierarchical Topology", "Topologie Hiérarchique"]:
            if st.session_state.language == 'fr':
                st.subheader("🏗️ Topologie Hiérarchique")
                st.info("Structure hiérarchique montrant les nœuds fog et les clients connectés.")
            else:
                st.subheader("🏗️ Hierarchical Topology")
                st.info("Hierarchical structure showing fog nodes and connected clients.")
        
        if not st.session_state.training_completed:
            if st.session_state.language == 'fr':
                st.info("Complétez l'entraînement fédéré pour voir les visualisations du réseau.")
            else:
                st.info("Complete federated training to see network visualizations.")

    elif st.session_state.current_tab_index == 7:
        # Tab 8: Advanced Analytics
        if st.session_state.language == 'fr':
            st.header("📊 Tableau de Bord Analytique Avancé")
        else:
            st.header("📊 Advanced Analytics Dashboard")
        
        if st.session_state.training_completed and st.session_state.advanced_analytics:
            st.session_state.advanced_analytics.create_medical_facility_dashboard()
        else:
            if st.session_state.language == 'fr':
                st.info("Complétez l'entraînement fédéré pour voir les analyses avancées.")
            else:
                st.info("Complete federated training to see advanced analytics.")

if __name__ == "__main__":
    main()