#!/usr/bin/env python3
"""
Convert JSON files from discomfort experiment to CSV format.

This script processes JSON files containing experiment data and creates two CSV files:
1. main_trials.csv - Contains survey-likert trial data with columns: subject_id, scenario_label, closeness_condition, action_0, action_1, action_2, action_3
2. exit_survey.csv - Contains survey-html-form trial data with columns: subject_id, gender, age, understood, comments, attention_passed, memory_correct_count
"""

import json
import csv
import os
import uuid
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

def process_json_files(input_dir, output_dir):
    """
    Process all JSON files in the input directory and create CSV files.
    
    Args:
        input_dir (str): Path to directory containing JSON files
        output_dir (str): Path to directory where CSV files will be saved
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lists to store data for each CSV
    main_trials_data = []
    exit_survey_data = []
    
    # Dictionary to map original subject IDs to anonymous IDs
    subject_id_mapping = {}
    
    def generate_deterministic_id(original_id):
        """Generate a deterministic UUID based on the original subject ID."""
        # Use a fixed namespace UUID for this project
        namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
        # Create a deterministic UUID based on the original subject ID
        return str(uuid.uuid5(namespace, original_id))
    
    # Process each JSON file
    json_files = list(input_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process")
    
    for json_file in json_files:
        print(f"Processing {json_file.name}...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract subject_id from the first trial
            original_subject_id = data[0].get('subject_id', 'unknown')
            
            # Create anonymous subject ID if not already mapped
            if original_subject_id not in subject_id_mapping:
                # Generate a deterministic UUID based on the original subject ID
                subject_id_mapping[original_subject_id] = generate_deterministic_id(original_subject_id)
            
            subject_id = subject_id_mapping[original_subject_id]
            
            # Process each trial in the data
            for trial in data:
                trial_type = trial.get('response_type', '')
                
                if trial_type == 'response':
                    # Extract main trial data
                    scenario_label = trial.get('scenario_label', '')
                    closeness_condition = trial.get('closeness_condition', '')
                    response = trial.get('response', {})
                    
                    # Extract action ratings
                    action_0 = response.get('action_0', '')
                    action_1 = response.get('action_1', '')
                    action_2 = response.get('action_2', '')
                    action_3 = response.get('action_3', '')
                    
                    main_trials_data.append({
                        'subject_id': subject_id,
                        'scenario_label': scenario_label,
                        'closeness_condition': closeness_condition,
                        'action_0': action_0,
                        'action_1': action_1,
                        'action_2': action_2,
                        'action_3': action_3,
                    })
                
                elif trial_type == 'exit_survey':
                    # Extract exit survey data
                    response = trial.get('response', {})
                    
                    gender = response.get('gender', '')
                    age = response.get('age', '')
                    understood = response.get('understood', '')
                    comments = response.get('comments', '')
                    attention_passed = trial.get('attention_passed', '')
                    memory_correct_count = trial.get('memory_correct_count', '')
                    exit_survey_data.append({
                        'subject_id': subject_id,
                        'gender': gender,
                        'age': age,
                        'understood': understood,
                        'comments': comments,
                        'attention_passed': attention_passed,
                        'memory_correct_count': memory_correct_count
                    })
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Write main trials CSV
    main_trials_file = output_path / 'main_trials.csv'
    with open(main_trials_file, 'w', newline='', encoding='utf-8') as f:
        if main_trials_data:
            fieldnames = ['subject_id', 'scenario_label', 'closeness_condition', 'action_0', 'action_1', 'action_2', 'action_3']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(main_trials_data)
            print(f"Created {main_trials_file} with {len(main_trials_data)} rows")
        else:
            print("No main trial data found")
    
    # Write exit survey CSV
    exit_survey_file = output_path / 'exit_survey.csv'
    with open(exit_survey_file, 'w', newline='', encoding='utf-8') as f:
        if exit_survey_data:
            fieldnames = ['subject_id', 'gender', 'age', 'understood', 'comments', 'attention_passed', 'memory_correct_count']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(exit_survey_data)
            print(f"Created {exit_survey_file} with {len(exit_survey_data)} rows")
        else:
            print("No exit survey data found")

def main():
    """Main function to run the conversion."""
    # Get project root directory
    project_root = get_project_root()
    
    # Define paths relative to project root
    input_dir = project_root / "data/discomfort/raw_data"
    output_dir = project_root / "data/discomfort"
    
    print("Converting JSON files to CSV...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process the files
    process_json_files(input_dir, output_dir)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
