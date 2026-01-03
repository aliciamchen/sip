#!/usr/bin/env python3
"""
Convert JSON files from experiments to CSV format.

This script processes JSON files containing experiment data and creates two CSV files:
1. main_trials.csv - Contains survey-likert trial data
2. exit_survey.csv - Contains survey-html-form trial data

Usage:
    python json_to_csv.py <experiment_name>
    
Available experiments:
    - forw_plan: Experiment with probability sliders for action ratings
    - inv_plan_intimacy: Inverse planning experiment measuring intimacy ratings before and after observing actions
    - inv_plan_reward: Inverse planning experiment measuring reward likelihood ratings before and after observing actions
"""

import json
import csv
import os
import uuid
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from utils import get_project_root

# Experiment configurations
EXPERIMENT_CONFIGS = {
    'forw_plan': {
        'description': 'Experiment with probability sliders for action ratings',
        'main_trial_fields': ['subject_id', 'scenario_label', 'intimacy_condition', 'reward_condition', 'action_0', 'action_1', 'action_2', 'action_3'],
        'exit_survey_fields': ['subject_id', 'gender', 'age', 'understood', 'comments', 'attention_passed', 'memory_correct_count'],
        'has_closeness': True,
        'has_attention_memory': True
    },
    'inv_plan_intimacy': {
        'description': 'Inverse planning experiment measuring intimacy ratings before and after observing actions',
        'main_trial_fields': ['subject_id', 'scenario_label', 'action_condition', 'reward_condition', 'stage', 'intimacy_rating'],
        'exit_survey_fields': ['subject_id', 'gender', 'age', 'understood', 'comments', 'attention_passed', 'memory_correct_count'],
        'has_closeness': False,
        'has_attention_memory': True
    },
    'inv_plan_reward': {
        'description': 'Inverse planning experiment measuring reward likelihood ratings before and after observing actions',
        'main_trial_fields': ['subject_id', 'scenario_label', 'action_condition', 'intimacy_condition', 'stage', 'response'],
        'exit_survey_fields': ['subject_id', 'gender', 'age', 'understood', 'comments', 'attention_passed', 'memory_correct_count'],
        'has_closeness': False,
        'has_attention_memory': True
    }
}

def process_json_files(input_dir, output_dir, config, experiment_name):
    """
    Process all JSON files in the input directory and create CSV files.
    
    Args:
        input_dir (str): Path to directory containing JSON files
        output_dir (str): Path to directory where CSV files will be saved
        config (dict): Experiment configuration
        experiment_name (str): Name of the experiment being processed
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
                    
                    # Handle different experiment types
                    if experiment_name == 'forw_plan':
                        # Extract action probabilities from probs field
                        probs = trial.get('probs', [])
                        action_0 = probs[0] if len(probs) > 0 else ''
                        action_1 = probs[1] if len(probs) > 1 else ''
                        action_2 = probs[2] if len(probs) > 2 else ''
                        action_3 = probs[3] if len(probs) > 3 else ''
                        
                        # Build trial data dictionary
                        trial_data = {
                            'subject_id': subject_id,
                            'scenario_label': scenario_label,
                            'action_0': action_0,
                            'action_1': action_1,
                            'action_2': action_2,
                            'action_3': action_3,
                        }
                        
                        # Add intimacy and reward conditions if this experiment has them
                        if config['has_closeness']:
                            trial_data['intimacy_condition'] = trial.get('intimacy_condition', '')
                            trial_data['reward_condition'] = trial.get('reward_condition', '')
                    
                    elif experiment_name == 'inv_plan_intimacy':
                        # Extract intimacy rating and stage information
                        intimacy_rating = trial.get('response', '')
                        stage = trial.get('stage', '')
                        action_condition = trial.get('action_condition', '')
                        reward_condition = trial.get('reward_condition', '')
                        
                        # Build trial data dictionary
                        trial_data = {
                            'subject_id': subject_id,
                            'scenario_label': scenario_label,
                            'action_condition': action_condition,
                            'reward_condition': reward_condition,
                            'stage': stage,
                            'intimacy_rating': intimacy_rating,
                        }
                    
                    elif experiment_name == 'inv_plan_reward':
                        # Extract reward likelihood rating and stage information
                        response = trial.get('response', '')
                        stage = trial.get('stage', '')
                        action_condition = trial.get('action_condition', '')
                        intimacy_condition = trial.get('intimacy_condition', '')
                        
                        # Build trial data dictionary
                        trial_data = {
                            'subject_id': subject_id,
                            'scenario_label': scenario_label,
                            'action_condition': action_condition,
                            'intimacy_condition': intimacy_condition,
                            'stage': stage,
                            'response': response,
                        }
                    
                    main_trials_data.append(trial_data)
                
                elif trial_type == 'exit_survey':
                    # Extract exit survey data
                    response = trial.get('response', {})
                    
                    gender = response.get('gender', '')
                    age = response.get('age', '')
                    understood = response.get('understood', '')
                    comments = response.get('comments', '')
                    
                    # Build survey data dictionary
                    survey_data = {
                        'subject_id': subject_id,
                        'gender': gender,
                        'age': age,
                        'understood': understood,
                        'comments': comments
                    }
                    
                    # Add attention and memory data if this experiment has it
                    if config['has_attention_memory']:
                        survey_data['attention_passed'] = trial.get('attention_passed', '')
                        survey_data['memory_correct_count'] = trial.get('memory_correct_count', '')
                    
                    exit_survey_data.append(survey_data)
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            continue
    
    # Write main trials CSV
    main_trials_file = output_path / 'main_trials.csv'
    with open(main_trials_file, 'w', newline='', encoding='utf-8') as f:
        if main_trials_data:
            writer = csv.DictWriter(f, fieldnames=config['main_trial_fields'])
            writer.writeheader()
            writer.writerows(main_trials_data)
            print(f"Created {main_trials_file} with {len(main_trials_data)} rows")
        else:
            print("No main trial data found")
    
    # Write exit survey CSV
    exit_survey_file = output_path / 'exit_survey.csv'
    with open(exit_survey_file, 'w', newline='', encoding='utf-8') as f:
        if exit_survey_data:
            writer = csv.DictWriter(f, fieldnames=config['exit_survey_fields'])
            writer.writeheader()
            writer.writerows(exit_survey_data)
            print(f"Created {exit_survey_file} with {len(exit_survey_data)} rows")
        else:
            print("No exit survey data found")

def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(
        description='Convert JSON files from experiments to CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available experiments:
  forw_plan         Experiment with probability sliders for action ratings
  inv_plan_intimacy Inverse planning experiment measuring intimacy ratings before and after observing actions
  inv_plan_reward   Inverse planning experiment measuring reward likelihood ratings before and after observing actions

Examples:
  python json_to_csv.py forw_plan
  python json_to_csv.py inv_plan_intimacy
  python json_to_csv.py inv_plan_reward
        """
    )
    
    parser.add_argument(
        'experiment',
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help='Name of the experiment to process'
    )
    
    args = parser.parse_args()
    
    # Get experiment configuration
    config = EXPERIMENT_CONFIGS[args.experiment]
    
    # Get project root directory
    project_root = get_project_root()
    
    # Define paths relative to project root
    input_dir = project_root / f"data/{args.experiment}/raw_data"
    output_dir = project_root / f"data/{args.experiment}"
    
    print(f"Converting JSON files to CSV for {args.experiment} experiment...")
    print(f"Description: {config['description']}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process the files
    process_json_files(input_dir, output_dir, config, args.experiment)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
