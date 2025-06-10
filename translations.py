"""
Multilingual translation module for the Hierarchical Federated Learning Platform
"""

# Translation dictionary with English and French
TRANSLATIONS = {
    'en': {
        # Page title and headers
        "page_title": "Hierarchical Federated Learning Platform",
        "sidebar_title": "🏥 FL Training Control",
        
        # Tab names
        "tab_training": "🚀 Training Control",
        "tab_monitoring": "🏥 Medical Station Monitoring", 
        "tab_visualization": "📊 Journey Visualization",
        "tab_analytics": "📈 Client Analytics",
        "tab_explainer": "🩺 Patient Risk Prediction Explainer",
        "tab_facility": "🏥 Advanced Medical Facility Analytics",
        "tab_risk": "🩺 Individual Patient Risk Assessment",
        
        # Training controls
        "model_type": "Model Type",
        "num_clients": "Number of Clients",
        "max_rounds": "Maximum Rounds",
        "target_accuracy": "Target Accuracy",
        "distribution_strategy": "Distribution Strategy",
        "aggregation_algorithm": "Aggregation Algorithm",
        "enable_privacy": "Enable Differential Privacy",
        "epsilon": "Epsilon (ε)",
        "delta": "Delta (δ)",
        "committee_size": "Committee Size",
        
        # Training interface sections
        "medical_network_config": "🏥 Medical Network Configuration",
        "num_medical_stations": "Number of Medical Stations", 
        "max_training_rounds": "Maximum Training Rounds",
        "model_selection": "🧠 Model Selection",
        "machine_learning_model": "Machine Learning Model",
        "fog_computing_setup": "🌫️ Fog Computing Setup",
        "enable_fog_nodes": "Enable Fog Nodes",
        "num_fog_nodes": "Number of Fog Nodes",
        "fog_aggregation_method": "Fog Aggregation Method",
        "privacy_configuration": "🔒 Privacy Configuration",
        "data_distribution": "📊 Data Distribution",
        
        # Buttons
        "start_training": "🚀 Start FL Training",
        "stop_training": "⏹️ Stop Training",
        "reset_training": "🔄 Reset Training",
        "new_session": "🔄 New Session",
        "analyze_risk": "Analyze Patient Risk",
        
        # Status messages
        "training_in_progress": "Training in progress...",
        "training_completed": "🎯 Federated Learning Training Completed Successfully!",
        "using_federated_model": "Using converged global federated model from completed training",
        "model_converged": "Model converged after {rounds} rounds with {accuracy:.3f} accuracy",
        "training_not_completed": "⚠️ Federated learning training not completed yet",
        "complete_training_first": "Please complete federated training first to use converged model for risk assessment",
        
        # Risk assessment
        "risk_assessment": "🎯 Risk Assessment",
        "risk_level": "Risk Level",
        "risk_score": "Risk Score", 
        "model_confidence": "Model Confidence",
        "low_risk": "Low Risk",
        "moderate_risk": "Moderate Risk",
        "high_risk": "High Risk",
        "very_high_risk": "Very High Risk",
        
        # Clinical guidance
        "clinical_guidance": "🏥 Clinical Guidance",
        "recommendation": "Recommendation",
        "continue_healthy_lifestyle": "Continue healthy lifestyle",
        "monitor_glucose": "Monitor glucose levels regularly",
        "consult_provider": "Consult healthcare provider soon",
        "immediate_attention": "Immediate medical attention recommended",
        
        # Patient information form
        "patient_information": "Patient Information",
        "pregnancies": "Number of Pregnancies",
        "glucose": "Glucose Level (mg/dL)",
        "blood_pressure": "Blood Pressure (mmHg)",
        "skin_thickness": "Skin Thickness (mm)",
        "insulin": "Insulin Level (μU/mL)",
        "bmi": "Body Mass Index (BMI)",
        "diabetes_pedigree": "Diabetes Pedigree Function",
        "age": "Age (years)",
        
        # Live Monitoring Tab
        "live_monitoring": "Live Monitoring",
        "training_progress": "Training Progress",
        "current_round": "Current Round",
        "rounds_completed": "Rounds Completed",
        "global_accuracy": "Global Accuracy",
        "active_medical_stations": "Active Medical Stations",
        "performance_metrics": "Performance Metrics",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "final_results": "Final Results",
        "protocol_type": "Protocol Type",
        "convergence": "Convergence",
        "converged": "Converged",
        "not_converged": "Not Converged",
        
        # Status messages
        "click_new_session": "Click 'New Session' to test different privacy budgets with fresh training",
        "training_medical_stations": "Training {num_clients} medical stations with {model_type} model...",
        "model_prediction_probability": "Model prediction probability: {score:.3f}",
        "model_prediction": "Model prediction: {score}",
        "federated_model_prediction_failed": "Federated model prediction failed: {error}",
        "training_may_not_be_completed": "Training may not be completed yet. Please run federated training first.",
        
        # Facility Analytics Tab
        "facility_analytics": "Facility Analytics",
        "medical_facility_dashboard": "Medical Facility Dashboard",
        "facility_overview": "Facility Overview",
        "performance_evolution": "Performance Evolution",
        "anomaly_detection": "Anomaly Detection",
        "convergence_analysis": "Convergence Analysis",
        
        # Journey Visualization Tab
        "journey_visualization": "Journey Visualization",
        "interactive_user_journey": "Interactive User Journey",
        "federated_learning_process": "Federated Learning Process",
        "data_flow": "Data Flow",
        "aggregation_process": "Aggregation Process",
        "pregnancies": "Number of Pregnancies",
        "glucose_level": "Glucose Level (mg/dL)",
        "blood_pressure": "Blood Pressure (mm Hg)",
        "skin_thickness": "Skin Thickness (mm)",
        "insulin": "Insulin (μU/mL)",
        "bmi": "BMI (kg/m²)",
        "diabetes_pedigree": "Diabetes Pedigree Function",
        "age": "Age (years)",
        
        # Help text
        "help_pregnancies": "Number of times pregnant",
        "help_glucose": "Plasma glucose concentration after 2 hours in oral glucose tolerance test",
        "help_blood_pressure": "Diastolic blood pressure",
        "help_skin_thickness": "Triceps skin fold thickness",
        "help_insulin": "2-Hour serum insulin",
        "help_bmi": "Body mass index",
        "help_diabetes_pedigree": "Diabetes pedigree function (genetic influence)",
        
        # Language selector
        "language_selector": "Language",
        "english": "English",
        "french": "Français"
    },
    
    'fr': {
        # Page title and headers
        "page_title": "Plateforme d'Apprentissage Fédéré Hiérarchique",
        "sidebar_title": "🏥 Contrôle de Formation FL",
        
        # Tab names
        "tab_training": "🚀 Contrôle d'Entraînement",
        "tab_monitoring": "🏥 Surveillance Station Médicale", 
        "tab_visualization": "📊 Visualisation du Parcours",
        "tab_analytics": "📈 Analytiques Client",
        "tab_explainer": "🩺 Explicateur de Prédiction de Risque Patient",
        "tab_facility": "🏥 Analytiques Avancées Établissement Médical",
        "tab_risk": "🩺 Évaluation Risque Patient Individuel",
        
        # Training controls
        "model_type": "Type de Modèle",
        "num_clients": "Nombre de Clients",
        "max_rounds": "Rondes Maximum",
        "target_accuracy": "Précision Cible",
        "distribution_strategy": "Stratégie de Distribution",
        "aggregation_algorithm": "Algorithme d'Agrégation",
        "enable_privacy": "Activer Confidentialité Différentielle",
        "epsilon": "Epsilon (ε)",
        "delta": "Delta (δ)",
        "committee_size": "Taille du Comité",
        
        # Training interface sections
        "medical_network_config": "🏥 Configuration Réseau Médical",
        "num_medical_stations": "Nombre de Stations Médicales", 
        "max_training_rounds": "Rondes d'Entraînement Maximum",
        "model_selection": "🧠 Sélection de Modèle",
        "machine_learning_model": "Modèle d'Apprentissage Automatique",
        "fog_computing_setup": "🌫️ Configuration Informatique Fog",
        "enable_fog_nodes": "Activer Nœuds Fog",
        "num_fog_nodes": "Nombre de Nœuds Fog",
        "fog_aggregation_method": "Méthode d'Agrégation Fog",
        "privacy_configuration": "🔒 Configuration de Confidentialité",
        "data_distribution": "📊 Distribution de Données",
        
        # Buttons
        "start_training": "🚀 Démarrer Formation FL",
        "stop_training": "⏹️ Arrêter Formation",
        "reset_training": "🔄 Réinitialiser Formation",
        "new_session": "🔄 Nouvelle Session",
        "analyze_risk": "Analyser Risque Patient",
        
        # Status messages
        "training_in_progress": "Formation en cours...",
        "training_completed": "🎯 Formation d'Apprentissage Fédéré Terminée avec Succès!",
        "using_federated_model": "Utilisation du modèle fédéré global convergé de formation terminée",
        "model_converged": "Modèle convergé après {rounds} rondes avec précision de {accuracy:.3f}",
        "training_not_completed": "⚠️ Formation d'apprentissage fédéré pas encore terminée",
        "complete_training_first": "Veuillez d'abord terminer la formation fédérée pour utiliser le modèle convergé pour l'évaluation des risques",
        
        # Risk assessment
        "risk_assessment": "🎯 Évaluation des Risques",
        "risk_level": "Niveau de Risque",
        "risk_score": "Score de Risque", 
        "model_confidence": "Confiance du Modèle",
        "low_risk": "Risque Faible",
        "moderate_risk": "Risque Modéré",
        "high_risk": "Risque Élevé",
        "very_high_risk": "Risque Très Élevé",
        
        # Clinical guidance
        "clinical_guidance": "🏥 Guidance Clinique",
        "recommendation": "Recommandation",
        "continue_healthy_lifestyle": "Continuer mode de vie sain",
        "monitor_glucose": "Surveiller régulièrement les niveaux de glucose",
        "consult_provider": "Consulter un professionnel de santé bientôt",
        "immediate_attention": "Attention médicale immédiate recommandée",
        
        # Patient information form
        "patient_information": "Informations Patient",
        "pregnancies": "Nombre de Grossesses",
        "glucose": "Niveau de Glucose (mg/dL)",
        "blood_pressure": "Pression Artérielle (mmHg)",
        "skin_thickness": "Épaisseur de la Peau (mm)",
        "insulin": "Niveau d'Insuline (μU/mL)",
        "bmi": "Indice de Masse Corporelle (IMC)",
        "diabetes_pedigree": "Fonction Pedigree Diabète",
        "age": "Âge (années)",
        
        # Live Monitoring Tab
        "live_monitoring": "Surveillance en Direct",
        "training_progress": "Progrès d'Entraînement",
        "current_round": "Ronde Actuelle",
        "rounds_completed": "Rondes Terminées",
        "global_accuracy": "Précision Globale",
        "active_medical_stations": "Stations Médicales Actives",
        "performance_metrics": "Métriques de Performance",
        "accuracy": "Précision",
        "precision": "Précision",
        "recall": "Rappel",
        "f1_score": "Score F1",
        "final_results": "Résultats Finaux",
        "protocol_type": "Type de Protocole",
        "convergence": "Convergence",
        "converged": "Convergé",
        "not_converged": "Non Convergé",
        
        # Status messages
        "click_new_session": "Cliquez sur 'Nouvelle Session' pour tester différents budgets de confidentialité avec un entraînement frais",
        "training_medical_stations": "Entraînement de {num_clients} stations médicales avec le modèle {model_type}...",
        "model_prediction_probability": "Probabilité de prédiction du modèle: {score:.3f}",
        "model_prediction": "Prédiction du modèle: {score}",
        "federated_model_prediction_failed": "Échec de la prédiction du modèle fédéré: {error}",
        "training_may_not_be_completed": "L'entraînement n'est peut-être pas encore terminé. Veuillez d'abord exécuter l'entraînement fédéré.",
        
        # Facility Analytics Tab
        "facility_analytics": "Analytiques d'Établissement",
        "medical_facility_dashboard": "Tableau de Bord Établissement Médical",
        "facility_overview": "Aperçu de l'Établissement",
        "performance_evolution": "Évolution des Performances",
        "anomaly_detection": "Détection d'Anomalies",
        "convergence_analysis": "Analyse de Convergence",
        
        # Journey Visualization Tab
        "journey_visualization": "Visualisation du Parcours",
        "interactive_user_journey": "Parcours Utilisateur Interactif",
        "federated_learning_process": "Processus d'Apprentissage Fédéré",
        "data_flow": "Flux de Données",
        "aggregation_process": "Processus d'Agrégation",
        "pregnancies": "Nombre de Grossesses",
        "glucose_level": "Niveau de Glucose (mg/dL)",
        "blood_pressure": "Pression Artérielle (mm Hg)",
        "skin_thickness": "Épaisseur de Peau (mm)",
        "insulin": "Insuline (μU/mL)",
        "bmi": "IMC (kg/m²)",
        "diabetes_pedigree": "Fonction Pedigree Diabète",
        "age": "Âge (années)",
        
        # Help text
        "help_pregnancies": "Nombre de fois enceinte",
        "help_glucose": "Concentration de glucose plasmatique après 2 heures dans le test de tolérance orale au glucose",
        "help_blood_pressure": "Pression artérielle diastolique",
        "help_skin_thickness": "Épaisseur du pli cutané triceps",
        "help_insulin": "Insuline sérique 2 heures",
        "help_bmi": "Indice de masse corporelle",
        "help_diabetes_pedigree": "Fonction pedigree diabète (influence génétique)",
        
        # Language selector
        "language_selector": "Langue",
        "english": "English",
        "french": "Français"
    }
}

def get_translation(key, lang='en', **kwargs):
    """Get translation for a given key and language"""
    translation = TRANSLATIONS.get(lang, {}).get(key, key)
    if kwargs:
        try:
            return translation.format(**kwargs)
        except:
            return translation
    return translation

def translate_risk_level(risk_score, lang='en'):
    """Translate risk level based on score"""
    if risk_score < 0.25:
        return get_translation("low_risk", lang)
    elif risk_score < 0.50:
        return get_translation("moderate_risk", lang)
    elif risk_score < 0.75:
        return get_translation("high_risk", lang)
    else:
        return get_translation("very_high_risk", lang)

def translate_clinical_advice(risk_score, lang='en'):
    """Translate clinical advice based on risk score"""
    if risk_score < 0.25:
        return get_translation("continue_healthy_lifestyle", lang)
    elif risk_score < 0.50:
        return get_translation("monitor_glucose", lang)
    elif risk_score < 0.75:
        return get_translation("consult_provider", lang)
    else:
        return get_translation("immediate_attention", lang)