#!/usr/bin/env python3
"""
Test script to verify medical facility distribution integration
"""

import sys
import os
sys.path.append('.')

from medical_facility_distribution import MedicalFacilityDistribution
import json

def test_file_load():
    """Test loading the saved distribution file"""
    
    print("Testing medical facility distribution file loading...")
    
    # Initialize distributor
    med_dist = MedicalFacilityDistribution(num_clients=8, random_state=42)
    
    # Try to load the saved file
    try:
        facility_info, success = med_dist.load_distribution_from_file("saved_medical_facility_distributions.txt")
        
        if success:
            print("✓ Successfully loaded distribution file")
            print(f"✓ Found {len(facility_info)} facilities")
            
            # Display loaded facilities
            for i, facility in enumerate(facility_info):
                print(f"  {i+1}. {facility['facility_type']}: {facility['total_patients']} patients, {facility['actual_diabetes_rate']*100:.1f}% diabetes rate")
            
            return facility_info
        else:
            print("✗ Failed to load distribution file")
            return None
            
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None

def verify_expected_distribution():
    """Verify the loaded distribution matches expected values"""
    
    facility_info = test_file_load()
    
    if facility_info:
        print("\nVerifying against expected distribution:")
        
        expected_facilities = [
            ("major_teaching_hospital", 136, 38.2),
            ("regional_medical_center", 68, 38.2),
            ("community_health_center", 20, 40.0),
            ("primary_care_clinic", 18, 38.9),
            ("specialty_diabetes_clinic", 16, 37.5),
            ("major_teaching_hospital", 14, 35.7),
            ("regional_medical_center", 13, 38.5),
            ("community_health_center", 11, 36.4)
        ]
        
        matches = 0
        for i, (exp_type, exp_patients, exp_rate) in enumerate(expected_facilities):
            if i < len(facility_info):
                actual = facility_info[i]
                actual_rate = actual['actual_diabetes_rate'] * 100
                
                if (actual['facility_type'] == exp_type and
                    actual['total_patients'] == exp_patients and
                    abs(actual_rate - exp_rate) < 0.1):
                    print(f"  ✓ Facility {i+1} matches: {exp_type}")
                    matches += 1
                else:
                    print(f"  ✗ Facility {i+1} mismatch:")
                    print(f"    Expected: {exp_type}, {exp_patients} patients, {exp_rate}% rate")
                    print(f"    Actual: {actual['facility_type']}, {actual['total_patients']} patients, {actual_rate:.1f}% rate")
        
        print(f"\nResult: {matches}/{len(expected_facilities)} facilities match exactly")
        return matches == len(expected_facilities)
    
    return False

if __name__ == "__main__":
    print("=== Medical Facility Distribution Integration Test ===")
    success = verify_expected_distribution()
    
    if success:
        print("\n🎉 All tests passed! Your uploaded distribution file is working correctly.")
    else:
        print("\n⚠️  Some tests failed. The integration may need adjustment.")