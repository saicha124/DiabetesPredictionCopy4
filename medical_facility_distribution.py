import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any
import random

class MedicalFacilityDistribution:
    """
    Realistic medical facility distribution for federated learning simulation
    Creates authentic healthcare facility types with realistic patient distributions
    """
    
    def __init__(self, num_clients: int = 9, random_state: int = 42):
        self.num_clients = num_clients
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define realistic medical facility types with expected characteristics
        self.facility_types = {
            'major_teaching_hospital': {
                'patient_range': (200, 400),
                'diabetes_rate_range': (0.32, 0.38),
                'complexity_factor': 1.2,  # More complex cases
                'priority': 1
            },
            'regional_medical_center': {
                'patient_range': (120, 250),
                'diabetes_rate_range': (0.30, 0.36),
                'complexity_factor': 1.1,
                'priority': 2
            },
            'community_hospital': {
                'patient_range': (80, 180),
                'diabetes_rate_range': (0.28, 0.34),
                'complexity_factor': 1.0,
                'priority': 3
            },
            'specialty_diabetes_clinic': {
                'patient_range': (40, 100),
                'diabetes_rate_range': (0.65, 0.85),  # Higher diabetes rate at specialty clinic
                'complexity_factor': 1.3,
                'priority': 4
            },
            'primary_care_clinic': {
                'patient_range': (30, 80),
                'diabetes_rate_range': (0.25, 0.32),
                'complexity_factor': 0.9,
                'priority': 5
            },
            'community_health_center': {
                'patient_range': (50, 120),
                'diabetes_rate_range': (0.35, 0.42),  # Higher rate due to underserved populations
                'complexity_factor': 1.0,
                'priority': 6
            },
            'urgent_care_center': {
                'patient_range': (25, 70),
                'diabetes_rate_range': (0.28, 0.35),
                'complexity_factor': 0.8,
                'priority': 7
            },
            'rural_health_clinic': {
                'patient_range': (20, 60),
                'diabetes_rate_range': (0.30, 0.38),
                'complexity_factor': 0.9,
                'priority': 8
            },
            'endocrinology_practice': {
                'patient_range': (30, 90),
                'diabetes_rate_range': (0.70, 0.90),  # Very high diabetes rate
                'complexity_factor': 1.4,
                'priority': 9
            }
        }
    
    def create_facility_assignments(self) -> List[Dict[str, Any]]:
        """Create unique facility assignments for each client"""
        
        # Sort facility types by priority to ensure balanced distribution
        sorted_facilities = sorted(
            self.facility_types.items(),
            key=lambda x: x[1]['priority']
        )
        
        facility_assignments = []
        
        # Assign unique facility types first
        for i in range(min(self.num_clients, len(sorted_facilities))):
            facility_name, facility_config = sorted_facilities[i]
            
            # Generate realistic patient count and diabetes rate
            patient_count = np.random.randint(
                facility_config['patient_range'][0],
                facility_config['patient_range'][1] + 1
            )
            
            diabetes_rate = np.random.uniform(
                facility_config['diabetes_rate_range'][0],
                facility_config['diabetes_rate_range'][1]
            )
            
            facility_assignments.append({
                'facility_id': i,
                'facility_type': facility_name,
                'patient_count': patient_count,
                'target_diabetes_rate': diabetes_rate,
                'complexity_factor': facility_config['complexity_factor']
            })
        
        # If we have more clients than facility types, create variations
        remaining_clients = self.num_clients - len(sorted_facilities)
        if remaining_clients > 0:
            for i in range(remaining_clients):
                # Select from existing facility types with variations
                base_facility_name, base_config = random.choice(sorted_facilities)
                
                # Create variation names
                variation_names = {
                    'major_teaching_hospital': f'university_medical_center_{i+2}',
                    'regional_medical_center': f'regional_hospital_{i+2}',
                    'community_hospital': f'community_hospital_{i+2}',
                    'primary_care_clinic': f'family_clinic_{i+2}',
                    'community_health_center': f'health_center_{i+2}'
                }
                
                facility_name = variation_names.get(base_facility_name, f'{base_facility_name}_{i+2}')
                
                # Add some variation to the characteristics
                patient_variance = 0.15  # 15% variance
                rate_variance = 0.05     # 5% variance
                
                base_patient_min, base_patient_max = base_config['patient_range']
                base_rate_min, base_rate_max = base_config['diabetes_rate_range']
                
                patient_count = np.random.randint(
                    max(15, int(base_patient_min * (1 - patient_variance))),
                    int(base_patient_max * (1 + patient_variance)) + 1
                )
                
                diabetes_rate = np.clip(
                    np.random.uniform(
                        base_rate_min * (1 - rate_variance),
                        base_rate_max * (1 + rate_variance)
                    ),
                    0.15, 0.95  # Reasonable bounds
                )
                
                facility_assignments.append({
                    'facility_id': len(sorted_facilities) + i,
                    'facility_type': facility_name,
                    'patient_count': patient_count,
                    'target_diabetes_rate': diabetes_rate,
                    'complexity_factor': base_config['complexity_factor']
                })
        
        return facility_assignments
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
        """
        Distribute data according to realistic medical facility characteristics
        
        Returns:
            - client_data: List of data dictionaries for each client
            - facility_info: List of facility information for each client
        """
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Cannot distribute empty dataset")
        
        facility_assignments = self.create_facility_assignments()
        
        # Calculate total target patients
        total_target_patients = sum(f['patient_count'] for f in facility_assignments)
        
        # If we have more target patients than actual data, scale down proportionally
        if total_target_patients > len(X):
            scale_factor = len(X) / total_target_patients
            for facility in facility_assignments:
                facility['patient_count'] = max(10, int(facility['patient_count'] * scale_factor))
        
        client_data = []
        facility_info = []
        
        # Track used indices to avoid duplicates
        used_indices = set()
        available_indices = list(range(len(X)))
        np.random.shuffle(available_indices)
        
        for facility in facility_assignments:
            target_patients = facility['patient_count']
            target_diabetes_rate = facility['target_diabetes_rate']
            
            # Calculate how many diabetic and non-diabetic patients we need
            target_diabetic = int(target_patients * target_diabetes_rate)
            target_non_diabetic = target_patients - target_diabetic
            
            # Get available diabetic and non-diabetic indices
            diabetic_indices = [i for i in available_indices if i not in used_indices and y[i] == 1]
            non_diabetic_indices = [i for i in available_indices if i not in used_indices and y[i] == 0]
            
            selected_indices = []
            
            # Select diabetic patients
            diabetic_to_select = min(target_diabetic, len(diabetic_indices))
            if diabetic_to_select > 0:
                selected_diabetic = np.random.choice(diabetic_indices, diabetic_to_select, replace=False)
                selected_indices.extend(selected_diabetic)
                used_indices.update(selected_diabetic)
            
            # Select non-diabetic patients
            non_diabetic_to_select = min(target_non_diabetic, len(non_diabetic_indices))
            if non_diabetic_to_select > 0:
                selected_non_diabetic = np.random.choice(non_diabetic_indices, non_diabetic_to_select, replace=False)
                selected_indices.extend(selected_non_diabetic)
                used_indices.update(selected_non_diabetic)
            
            # If we couldn't get enough patients of the target distribution, fill with any available
            while len(selected_indices) < min(target_patients, len(available_indices) - len(used_indices)):
                remaining_indices = [i for i in available_indices if i not in used_indices]
                if not remaining_indices:
                    break
                additional_index = np.random.choice(remaining_indices)
                selected_indices.append(additional_index)
                used_indices.add(additional_index)
            
            # Ensure minimum data per client
            if len(selected_indices) < 5:
                remaining_indices = [i for i in available_indices if i not in used_indices]
                additional_needed = min(5 - len(selected_indices), len(remaining_indices))
                if additional_needed > 0:
                    additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False)
                    selected_indices.extend(additional_indices)
                    used_indices.update(additional_indices)
            
            if len(selected_indices) == 0:
                # Emergency fallback - give at least one sample
                if available_indices:
                    selected_indices = [available_indices[0]]
                    used_indices.add(available_indices[0])
            
            # Extract data for this facility
            facility_X = X[selected_indices]
            facility_y = y[selected_indices]
            
            # Create train/test split
            if len(facility_X) > 1:
                if len(np.unique(facility_y)) > 1:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            facility_X, facility_y, 
                            test_size=0.2, 
                            random_state=self.random_state + facility['facility_id'],
                            stratify=facility_y
                        )
                    except ValueError:
                        # Fallback to simple split
                        split_idx = max(1, int(0.8 * len(facility_X)))
                        X_train, X_test = facility_X[:split_idx], facility_X[split_idx:]
                        y_train, y_test = facility_y[:split_idx], facility_y[split_idx:]
                else:
                    split_idx = max(1, int(0.8 * len(facility_X)))
                    X_train, X_test = facility_X[:split_idx], facility_X[split_idx:]
                    y_train, y_test = facility_y[:split_idx], facility_y[split_idx:]
            else:
                X_train = X_test = facility_X
                y_train = y_test = facility_y
            
            # Store client data
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
            
            # Calculate actual diabetes rate
            all_y = np.concatenate([y_train, y_test]) if len(y_train) > 0 and len(y_test) > 0 else facility_y
            actual_diabetes_rate = np.mean(all_y) if len(all_y) > 0 else 0
            
            # Store facility information
            facility_info.append({
                'facility_id': facility['facility_id'],
                'facility_type': facility['facility_type'],
                'total_patients': len(selected_indices),
                'target_diabetes_rate': target_diabetes_rate,
                'actual_diabetes_rate': actual_diabetes_rate,
                'complexity_factor': facility['complexity_factor'],
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
        
        return client_data, facility_info
    
    def get_distribution_summary(self, facility_info: List[Dict[str, Any]]) -> str:
        """Generate a summary of the facility distribution"""
        
        summary = "🏥 Authentic Medical Facility Distribution:\n\n"
        
        # Sort by total patients (descending)
        sorted_facilities = sorted(facility_info, key=lambda x: x['total_patients'], reverse=True)
        
        for facility in sorted_facilities:
            diabetes_rate_pct = facility['actual_diabetes_rate'] * 100
            summary += f"• {facility['facility_type']}: {facility['total_patients']} patients, "
            summary += f"{diabetes_rate_pct:.1f}% diabetes prevalence\n"
        
        # Add summary statistics
        total_patients = sum(f['total_patients'] for f in facility_info)
        avg_diabetes_rate = np.mean([f['actual_diabetes_rate'] for f in facility_info]) * 100
        
        summary += f"\n📊 Distribution Summary:\n"
        summary += f"• Total Patients: {total_patients}\n"
        summary += f"• Average Diabetes Rate: {avg_diabetes_rate:.1f}%\n"
        summary += f"• Number of Facilities: {len(facility_info)} (all unique)\n"
        
        return summary