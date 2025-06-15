import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import io
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import modules
from translations import get_translation
from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from differential_privacy import DifferentialPrivacyManager
from data_distribution import get_distribution_strategy
from client_simulator import ClientSimulator
from journey_visualization import InteractiveJourneyVisualizer
from performance_optimizer import FederatedLearningOptimizer
from real_medical_data_fetcher import RealMedicalDataFetcher, load_authentic_medical_data
from advanced_client_analytics import AdvancedClientAnalytics
from client_visualization import ClientPerformanceVisualizer


def create_performance_optimizer():
    """Create and return a performance optimizer instance"""
    return FederatedLearningOptimizer()


def init_session_state():
    """Initialize session state variables"""
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'best_accuracy' not in st.session_state:
        st.session_state.best_accuracy = 0.0
    if 'global_model_accuracy' not in st.session_state:
        st.session_state.global_model_accuracy = 0.0
    if 'journey_visualizer' not in st.session_state:
        st.session_state.journey_visualizer = InteractiveJourneyVisualizer()
    if 'analytics_engine' not in st.session_state:
        st.session_state.analytics_engine = AdvancedClientAnalytics()
    if 'client_visualizer' not in st.session_state:
        st.session_state.client_visualizer = ClientPerformanceVisualizer()
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0


def main():
    st.set_page_config(
        page_title=get_translation("app_title", st.session_state.get('language', 'en')),
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header with language selector
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        st.title("🏥 " + get_translation("app_title", st.session_state.language))
        st.markdown(get_translation("app_description", st.session_state.language))
    
    with col3:
        # Language selector
        current_lang = "🇫🇷 Français" if st.session_state.language == 'fr' else "🇺🇸 English"
        if st.button(current_lang, key="lang_selector"):
            st.session_state.language = 'fr' if st.session_state.language == 'en' else 'en'
            st.rerun()
    
    st.markdown("---")
    
    # Data loading section
    if not st.session_state.data_loaded:
        with st.expander(get_translation("load_medical_data", st.session_state.language), expanded=True):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(get_translation("authentic_medical_data_info", st.session_state.language))
            
            with col2:
                if st.button(get_translation("load_diabetes_dataset", st.session_state.language), type="primary"):
                    with st.spinner(get_translation("loading_authentic_data", st.session_state.language)):
                        try:
                            # Load authentic medical data
                            data = load_authentic_medical_data()
                            st.session_state.data = data
                            st.session_state.data_loaded = True
                            st.success(get_translation("data_loaded_successfully", st.session_state.language, samples=len(data)))
                            st.rerun()
                        except Exception as e:
                            st.error(get_translation("failed_to_load_dataset", st.session_state.language, error=str(e)))
                            return

    # Tab navigation with arrows
    col1, col2, col3 = st.columns([1, 6, 1])
    
    tab_names = [
        get_translation("tab_training", st.session_state.language),
        get_translation("tab_monitoring", st.session_state.language), 
        get_translation("tab_visualization", st.session_state.language),
        get_translation("tab_analytics", st.session_state.language),
        get_translation("tab_facility", st.session_state.language),
        get_translation("tab_risk", st.session_state.language),
        get_translation("tab_graph_viz", st.session_state.language),
        get_translation("tab_advanced_analytics", st.session_state.language)
    ]
    
    with col1:
        if st.button("◀ Précédent" if st.session_state.language == 'fr' else "◀ Previous", key="prev_tab"):
            if st.session_state.active_tab > 0:
                st.session_state.active_tab -= 1
                st.rerun()
    
    with col2:
        current_tab_name = tab_names[st.session_state.active_tab]
        if st.session_state.language == 'fr':
            st.markdown(f"**Onglet actuel:** {current_tab_name} ({st.session_state.active_tab + 1}/8)")
        else:
            st.markdown(f"**Current tab:** {current_tab_name} ({st.session_state.active_tab + 1}/8)")
    
    with col3:
        if st.button("Suivant ▶" if st.session_state.language == 'fr' else "Next ▶", key="next_tab"):
            if st.session_state.active_tab < len(tab_names) - 1:
                st.session_state.active_tab += 1
                st.rerun()

    # Display content based on active tab
    st.markdown("---")
    
    if st.session_state.active_tab == 0:  # Training tab
        st.header("🎛️ " + get_translation("tab_training", st.session_state.language))
        
        if st.session_state.data_loaded:
            st.success("Training interface loaded successfully")
        else:
            st.warning(get_translation("load_data_first", st.session_state.language))
    
    elif st.session_state.active_tab == 1:  # Monitoring tab
        st.header("📊 " + get_translation("tab_monitoring", st.session_state.language))
        
        if st.session_state.training_started:
            st.info("Training monitoring interface")
        else:
            st.warning(get_translation("start_training_to_monitor", st.session_state.language))

    elif st.session_state.active_tab == 2:  # Visualization tab
        st.header("🗺️ " + get_translation("tab_visualization", st.session_state.language))
        st.info("Interactive learning journey visualization")

    elif st.session_state.active_tab == 3:  # Analytics tab
        st.header("📊 " + get_translation("tab_analytics", st.session_state.language))
        st.info("Performance analysis dashboard")

    elif st.session_state.active_tab == 4:  # Facility tab
        st.header("🏥 " + get_translation("tab_facility", st.session_state.language))
        st.info("Medical facility analytics")

    elif st.session_state.active_tab == 5:  # Risk tab
        st.header("🩺 " + get_translation("tab_risk", st.session_state.language))
        st.info("Patient risk assessment")

    elif st.session_state.active_tab == 6:  # Graph visualization tab
        st.header("📈 " + get_translation("tab_graph_viz", st.session_state.language))
        st.info("Advanced graph visualizations")

    elif st.session_state.active_tab == 7:  # Advanced analytics tab
        st.header("🔬 " + get_translation("tab_advanced_analytics", st.session_state.language))
        st.info("Advanced analytics dashboard")


if __name__ == "__main__":
    main()