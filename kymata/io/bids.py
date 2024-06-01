import os
import json
import shutil
import pandas as pd

# Mapping dictionary
participant_mapping = {
    'pilot_01': 'E1',
    'pilot_02': 'E2',
    'participant_01': 'E3',
    'participant_01b': 'E3b',
    'participant_02': 'E4',
    'participant_03': 'E5',
    'participant_04': 'E6',
    'participant_05': 'E7',
    'participant_07': 'E8',
    'participant_08': 'E9',
    'participant_09': 'E10',
    'participant_10': 'E11',
    'participant_11': 'E12',
    'participant_12': 'E13',
    'participant_13': 'E14',
    'participant_14': 'E15',
    'participant_15': 'E16',
    'participant_16': 'E17',
    'participant_17': 'E18',
    'participant_18': 'E19',
    'participant_19': 'E20'
}

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_meg_to_bids(participant, new_id, input_base_path, output_base_path):
    participant_id = f'sub-{new_id}'
    subj_dir = os.path.join(output_base_path, participant_id)
    meg_dir = os.path.join(subj_dir, 'meg')
    create_dirs(meg_dir)

    # Iterate over files in participant directory
    participant_path = os.path.join(input_base_path, participant)
    for file in os.listdir(participant_path):
        if file.endswith('.fif'):
            task_name = 'rest'
            if 'empty_room' in file:
                task_name = 'emptyroom'
            elif 'run' in file:
                run_number = file.split('_')[2].replace('run', '').replace('.fif', '')
                task_name = f'task-run{run_number}'

            # Determine new filename
            new_filename = f'{participant_id}_{task_name}_meg.fif'
            src = os.path.join(participant_path, file)
            dest = os.path.join(meg_dir, new_filename)
            shutil.copy(src, dest)

            # Create corresponding metadata files
            create_metadata_files(participant_id, task_name, meg_dir)

def create_metadata_files(participant_id, task_name, meg_dir):
    meg_json = {
        "TaskName": task_name,
        "SamplingFrequency": 1000.0,  # Update this with actual value
        "PowerLineFrequency": 50,  # Update this with actual value
        "DewarPosition": "upright",
        "SoftwareFilters": "n/a",
        "DigitizedLandmarks": True,
        "DigitizedHeadPoints": True
    }
    channels_tsv = pd.DataFrame({
        "name": ["MEG0113", "MEG0112"],  # Update with actual channel names
        "type": ["MEGGRAD", "MEGGRAD"],  # Update with actual channel types
        "units": ["T/m", "T/m"],
        "low_cutoff": [0.1, 0.1],  # Update with actual values
        "high_cutoff": [330, 330],  # Update with actual values
        "description": ["Channel description", "Channel description"]
    })
    coordsystem_json = {
        "MEGCoordinateSystem": "CTF",
        "MEGCoordinateUnits": "cm",
        "HeadCoilCoordinates": {
            "NAS": [0.0, 0.0915, -0.0334],
            "LPA": [-0.0731, 0.0004, -0.0419],
            "RPA": [0.0731, 0.0004, -0.0419]
        }
    }
    events_tsv = pd.DataFrame({
        "onset": [0.0],
        "duration": [1.0],
        "trial_type": ["rest"]
    })

    # Write metadata files
    meg_json_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_meg.json')
    channels_tsv_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_channels.tsv')
    coordsystem_json_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_coordsystem.json')
    events_tsv_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_events.tsv')

    with open(meg_json_path, 'w') as f:
        json.dump(meg_json, f, indent=4)
    channels_tsv.to_csv(channels_tsv_path, sep='\t', index=False)
    with open(coordsystem_json_path, 'w') as f:
        json.dump(coordsystem_json, f, indent=4)
    events_tsv.to_csv(events_tsv_path, sep='\t', index=False)
