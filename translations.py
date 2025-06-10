"""
French translation module for the Hierarchical Federated Learning Platform
"""

# Main interface translations
TRANSLATIONS = {
    # Page title and headers
    "page_title": "Plateforme d'Apprentissage Fédéré Hiérarchique",
    "sidebar_title": "🏥 Contrôle de Formation FL",
    
    # Tab names
    "tab_training": "🚀 Contrôle de Formation",
    "tab_monitoring": "🏥 Surveillance Station Médicale", 
    "tab_visualization": "📊 Visualisation Parcours",
    "tab_analytics": "📈 Analyses Clients",
    "tab_explainer": "🩺 Explicateur Prédiction Risque Patient",
    "tab_facility": "🏥 Analyses Avancées Établissement Médical",
    "tab_risk": "🩺 Évaluation Risque Patient Individuel",
    
    # Training controls
    "model_type": "Type de Modèle",
    "num_clients": "Nombre de Clients",
    "max_rounds": "Rounds Maximum",
    "target_accuracy": "Précision Cible",
    "distribution_strategy": "Stratégie Distribution",
    "aggregation_algorithm": "Algorithme Agrégation",
    "enable_privacy": "Activer Confidentialité Différentielle",
    "epsilon": "Epsilon (ε)",
    "delta": "Delta (δ)",
    "committee_size": "Taille Comité",
    
    # Buttons
    "start_training": "🚀 Démarrer Formation FL",
    "stop_training": "⏹️ Arrêter Formation",
    "reset_training": "🔄 Réinitialiser Formation",
    "new_session": "🔄 Nouvelle Session",
    "analyze_risk": "🔍 Analyser Risque Patient",
    
    # Status messages
    "training_in_progress": "Formation en cours...",
    "training_completed": "🎯 Formation Apprentissage Fédéré Terminée avec Succès!",
    "using_federated_model": "✅ Utilisation du modèle fédéré global convergé de formation terminée",
    "model_converged": "Modèle convergé après {rounds} rounds avec {accuracy:.3f} précision",
    "training_not_completed": "⚠️ Formation apprentissage fédéré pas encore terminée",
    "complete_training_first": "Veuillez terminer la formation fédérée d'abord pour utiliser le modèle convergé pour l'évaluation des risques",
    
    # Risk assessment
    "risk_assessment": "🎯 Évaluation Risque",
    "risk_level": "Niveau de Risque",
    "risk_score": "Score de Risque", 
    "model_confidence": "Confiance du Modèle",
    "low_risk": "Risque Faible",
    "moderate_risk": "Risque Modéré",
    "high_risk": "Risque Élevé",
    "very_high_risk": "Risque Très Élevé",
    
    # Clinical guidance
    "clinical_guidance": "🏥 Orientation Clinique",
    "recommendation": "Recommandation",
    "continue_healthy_lifestyle": "Continuer mode de vie sain",
    "monitor_glucose": "Surveiller niveaux glucose régulièrement",
    "consult_provider": "Consulter professionnel santé bientôt",
    "immediate_attention": "Attention médicale immédiate recommandée",
    
    # Patient information form
    "patient_information": "Informations Patient",
    "pregnancies": "Nombre de Grossesses",
    "glucose_level": "Niveau Glucose (mg/dL)",
    "blood_pressure": "Tension Artérielle (mm Hg)",
    "skin_thickness": "Épaisseur Peau (mm)",
    "insulin": "Insuline (μU/mL)",
    "bmi": "IMC (kg/m²)",
    "diabetes_pedigree": "Fonction Pedigree Diabète",
    "age": "Âge (années)",
    
    # Help text
    "help_pregnancies": "Nombre de fois enceinte",
    "help_glucose": "Concentration glucose plasma après 2 heures test tolérance glucose oral",
    "help_blood_pressure": "Tension artérielle diastolique",
    "help_skin_thickness": "Épaisseur pli cutané triceps",
    "help_insulin": "Insuline sérique 2 heures",
    "help_bmi": "Indice masse corporelle",
    "help_diabetes_pedigree": "Fonction pedigree diabète (influence génétique)",
    
    # Performance metrics
    "accuracy": "Précision",
    "loss": "Perte",
    "f1_score": "Score F1",
    "precision": "Précision",
    "recall": "Rappel",
    "rounds_completed": "Rounds Terminés",
    "training_time": "Temps Formation",
    "convergence_status": "Statut Convergence",
    
    # Medical facilities
    "major_teaching_hospital": "Hôpital Universitaire Principal",
    "regional_medical_center": "Centre Médical Régional",
    "community_health_center": "Centre Santé Communautaire",
    "specialized_diabetes_clinic": "Clinique Diabète Spécialisée",
    "rural_health_facility": "Établissement Santé Rural",
    
    # Privacy and security
    "differential_privacy": "Confidentialité Différentielle",
    "committee_validation": "Validation Comité",
    "secret_sharing": "Partage Secret",
    "privacy_budget": "Budget Confidentialité",
    "noise_scale": "Échelle Bruit",
    
    # Feature analysis
    "feature_importance": "Importance Caractéristiques",
    "feature_contributions": "Contributions Caractéristiques au Risque",
    "normal_ranges": "Plages Normales",
    "risk_factors": "Facteurs de Risque",
    "protective_factors": "Facteurs Protecteurs",
    
    # Error messages
    "training_failed": "Formation échouée",
    "import_error": "Erreur d'importation",
    "preprocessing_error": "Erreur prétraitement",
    "model_prediction_failed": "Prédiction modèle fédéré échouée",
    "training_not_ready": "Formation peut ne pas être terminée encore. Veuillez exécuter formation fédérée d'abord.",
    
    # Data information
    "authentic_medical_data": "Données Médicales Authentiques",
    "patients_loaded": "patients chargés",
    "diabetes_prevalence": "prévalence diabète",
    "data_preprocessed": "Données prétraitées",
    "samples": "échantillons",
    "features": "caractéristiques",
    
    # Visualization
    "performance_evolution": "Évolution Performance",
    "client_comparison": "Comparaison Clients",
    "round_analysis": "Analyse Round",
    "confusion_matrix": "Matrice Confusion",
    "prediction_distribution": "Distribution Prédictions",
    "convergence_analysis": "Analyse Convergence",
    
    # Clinical thresholds and ranges
    "fasting_glucose_diabetic": "Glucose à jeun ≥126 mg/dL (plage diabétique)",
    "fasting_glucose_prediabetic": "Glucose à jeun 100-125 mg/dL (prédiabétique)",
    "normal_glucose": "Niveaux glucose normaux",
    "obesity_bmi": "Obésité (IMC ≥30)",
    "overweight_bmi": "Surpoids (IMC 25-30)",
    "normal_weight": "Poids normal",
    "advanced_age": "Âge avancé (≥45 ans)",
    "family_history": "Antécédents familiaux significatifs",
    
    # Population comparison
    "population_comparison": "Comparaison Population",
    "patient_percentile": "Le risque du patient est plus élevé que {percentile:.1f}% de la population",
    
    # Model information
    "model_prediction_probability": "Probabilité prédiction modèle: {probability:.3f}",
    "model_prediction": "Prédiction modèle: {prediction}",
    "using_statistical_model": "Utilisation modèle statistique pour prédiction",
    
    # Advanced analytics
    "facility_overview": "Aperçu Établissement",
    "performance_metrics": "Métriques Performance",
    "anomaly_detection": "Détection Anomalies",
    "underperforming_facilities": "Établissements sous-performants",
    "performance_outliers": "Valeurs aberrantes performance",
    
    # Journey visualization
    "interactive_journey": "Parcours Interactif",
    "patient_flow": "Flux Patients",
    "decision_points": "Points Décision",
    "treatment_pathways": "Voies Traitement",
    
    # System status
    "system_ready": "Système Prêt",
    "loading": "Chargement...",
    "processing": "Traitement...",
    "complete": "Terminé",
    "failed": "Échoué",
    
    # Additional clinical terms
    "risk_meter": "📊 Compteur Risque",
    "clinical_interpretation": "🏥 Interprétation Clinique",
    "risk_detected": "**Risque diabète élevé détecté**",
    "moderate_risk_detected": "**Risque diabète modéré détecté**",
    "low_risk_detected": "**Risque diabète faible détecté**"
}

def get_translation(key, **kwargs):
    """Get translated text with optional formatting"""
    text = TRANSLATIONS.get(key, key)
    if kwargs:
        try:
            return text.format(**kwargs)
        except (KeyError, ValueError):
            return text
    return text

def translate_risk_level(risk_score):
    """Translate risk level based on score"""
    if risk_score < 0.25:
        return get_translation("low_risk")
    elif risk_score < 0.50:
        return get_translation("moderate_risk")
    elif risk_score < 0.75:
        return get_translation("high_risk")
    else:
        return get_translation("very_high_risk")

def translate_clinical_advice(risk_score):
    """Translate clinical advice based on risk score"""
    if risk_score < 0.25:
        return get_translation("continue_healthy_lifestyle")
    elif risk_score < 0.50:
        return get_translation("monitor_glucose")
    elif risk_score < 0.75:
        return get_translation("consult_provider")
    else:
        return get_translation("immediate_attention")