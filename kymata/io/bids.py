import os
import json
import shutil
import pandas as pd
import mne
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

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

def convert_meg_to_bids(participant, new_id, task, input_base_path, output_base_path):
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
                task_name = f'task-{task}_run-{run_number}'

            # Determine new filename
            new_filename = f'{participant_id}_{task_name}_meg.fif'
            src = os.path.join(participant_path, file)
            dest = os.path.join(meg_dir, new_filename)
            shutil.copy(src, dest)

            # Create corresponding metadata files
            create_metadata_files(participant_id, task_name, src, meg_dir)

def create_metadata_files(participant_id, task_name, src, meg_dir):

    raw = mne.io.read_raw_fif(src, preload=False)

    meg_json = get_meg_info(raw, task_name)

    channels_tsv = get_channel_info(raw)

    if task_name != 'emptyroom':
        coordsystem_json = get_coordinate_info(raw)
    
    events_tsv = get_event_info(raw)

    # Write metadata files
    meg_json_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_meg.json')
    channels_tsv_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_channels.tsv')
    if task_name != 'emptyroom':
        coordsystem_json_path = os.path.join(meg_dir, f'{participant_id}_coordsystem.json')
    events_tsv_path = os.path.join(meg_dir, f'{participant_id}_{task_name}_events.tsv')

    with open(meg_json_path, 'w') as f:
        json.dump(meg_json, f, indent=4)
    channels_tsv.to_csv(channels_tsv_path, sep='\t', index=False)
    if task_name != 'emptyroom':
        with open(coordsystem_json_path, 'w') as f:
            json.dump(coordsystem_json, f, indent=4)
    events_tsv.to_csv(events_tsv_path, sep='\t', index=False)

def get_coordinate_info(raw):

    # Initialize the output structure
    coordinate_info = {
        "MEGCoordinateSystem": None,
        "MEGCoordinateUnits": "mm",  # Assuming meters if not specified
        "MEGCoordinateSystemDescription": None,
        "HeadCoilCoordinates": {},
        "HeadCoilCoordinateSystem": None,
        "HeadCoilCoordinateUnits": "mm",  # Assuming meters if not specified
        "AnatomicalLandmarkCoordinates": {},
        "AnatomicalLandmarkCoordinateSystem": None,
        "AnatomicalLandmarkCoordinateUnits": "mm"  # Assuming meters if not specified
    }

    # Extract MEG and coordinate system information
    meg_system = raw.info.get('dig')
    if meg_system:
        # This can vary; often found in 'coord_frame'
        coord_frame = meg_system[0]['coord_frame']
        if coord_frame == 4:
            coordinate_info["MEGCoordinateSystem"] = "CTF"
            coordinate_info["MEGCoordinateSystemDescription"] = "CTF/4D system"
        elif coord_frame == 5:
            coordinate_info["MEGCoordinateSystem"] = "ITRS"
            coordinate_info["MEGCoordinateSystemDescription"] = "Internal Target Reference System"
        elif coord_frame == 6:
            coordinate_info["MEGCoordinateSystem"] = "ALS"
            coordinate_info["MEGCoordinateSystemDescription"] = "ALS orientation and the origin between the ears"
        # Add more cases as needed

    # Extract head coil (HPI) coordinates
    hpi_info = raw.info.get('hpi_results')
    if hpi_info:
        hpi_dig_points = hpi_info[0].get('dig_points', [])
        for i, coil in enumerate(hpi_dig_points, start=1):
            coordinate_info["HeadCoilCoordinates"][f"coil{i}"] = coil['r'].tolist()

    # Extract anatomical landmarks (fiducials) coordinates
    dig = raw.info.get('dig')
    landmark_mapping = {1: "NAS", 2: "LPA", 3: "RPA"}  # Mapping for fiducials
    for point in dig:
        if point['kind'] == 1:  # Kind 1 corresponds to the fiducials
            ident = point['ident']
            if ident in landmark_mapping:
                coordinate_info["AnatomicalLandmarkCoordinates"][landmark_mapping[ident]] = point['r'].tolist()

    # Fill anatomical landmark coordinates in both HeadCoilCoordinates and AnatomicalLandmarkCoordinates
    for key, value in coordinate_info["AnatomicalLandmarkCoordinates"].items():
        coordinate_info["HeadCoilCoordinates"][key] = value

    # Assuming the coordinate systems for head coils and anatomical landmarks are the same as MEG
    coordinate_info["HeadCoilCoordinateSystem"] = coordinate_info["MEGCoordinateSystem"]
    coordinate_info["AnatomicalLandmarkCoordinateSystem"] = coordinate_info["MEGCoordinateSystem"]

    return coordinate_info

def get_meg_info(raw, task_name):

    meg_json = {
        "TaskName": "0",  # Task information may need to be specified manually
        "Manufacturer": raw.info.get('manufacturer', 'Elekta Neuromag'),
        "PowerLineFrequency": raw.info.get('line_freq', 'n/a'),
        "SamplingFrequency": raw.info.get('sfreq', 'n/a'),
        "SoftwareFilters": {
            "SpatialCompensation": {
                "GradientOrder": 0  # This may need to be set based on specific info
            }
        },
        "RecordingDuration": raw.times[-1] if raw.times.size > 0 else 0,
        "RecordingType": "continuous",  # Typically continuous for MEG
        "DewarPosition": "n/a",  # This info might not be directly available
        "DigitizedLandmarks": any(p['kind'] == 1 for p in raw.info['dig']) if raw.info.get('dig') else False,
        "DigitizedHeadPoints": any(p['kind'] == 2 for p in raw.info['dig']) if raw.info.get('dig') else False,
        "MEGChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_MEG_CH),
        "MEGREFChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_REF_MEG_CH),
        "ContinuousHeadLocalization": True,
        "HeadCoilFrequency": [raw.info.get('hpi_meas', [{}])[0].get('hpi_coils', {})[i].get('coil_freq', []) for i in range(5)] if task_name != 'emptyroom' else [],
        "EEGChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EEG_CH),
        "EOGChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EOG_CH),
        "ECGChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_ECG_CH),
        "EMGChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_EMG_CH),
        "MiscChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_MISC_CH),
        "TriggerChannelCount": sum(1 for ch in raw.info['chs'] if ch['kind'] == mne.io.constants.FIFF.FIFFV_STIM_CH)
    }

    return meg_json

def get_channel_info(raw):

    # Initialize the list to hold channel information
    channels_info = []

    # Iterate through the channels in the FIF file
    for idx, ch in enumerate(raw.info['chs']):
        ch_name = ch['ch_name']
        ch_type = raw.get_channel_types()[idx].upper()  # Get the channel type and capitalize it
        
        # Define channel types for BIDS
        if ch_type == 'MAG':
            ch_type_bids = 'MEGMAG'
            units = 'T'  # Tesla for magnetometers
        elif ch_type == 'GRAD':
            ch_type_bids = 'MEGGRADAXIAL'
            units = 'T/m'  # Tesla per meter for gradiometers
        elif ch_type == 'EEG':
            ch_type_bids = 'EEG'
            units = 'uV'  # Microvolts for EEG
        elif ch_type == 'EOG':
            ch_type_bids = 'EOG'
            units = 'uV'  # Microvolts for EOG
        elif ch_type == 'ECG':
            ch_type_bids = 'ECG'
            units = 'uV'  # Microvolts for ECG
        elif ch_type == 'EMG':
            ch_type_bids = 'EMG'
            units = 'uV'  # Microvolts for EMG
        elif ch_type == 'STIM':
            ch_type_bids = 'TRIGGER'
            units = 'n/a'  # No units for trigger channels
        else:
            ch_type_bids = 'MISC'
            units = 'n/a'  # Miscellaneous channels may not have units

        # Extract low and high pass filter information, if available
        low_cutoff = ch['loc'][3] if ch['loc'][3] > 0 else 0.0
        high_cutoff = ch['loc'][4] if ch['loc'][4] > 0 else raw.info['sfreq'] / 2

        # Default channel status
        status = 'good'
        status_description = 'n/a'

        # Append the channel information to the list
        channels_info.append([
            ch_name,
            ch_type_bids,
            units,
            low_cutoff,
            high_cutoff,
            ch_type,
            raw.info['sfreq'],
            status,
            status_description
        ])

    # Create a DataFrame from the list
    columns = [
        "name", "type", "units", "low_cutoff", "high_cutoff", 
        "description", "sampling_frequency", "status", "status_description"
    ]
    channels_df = pd.DataFrame(channels_info, columns=columns)

    return channels_df

def get_event_info(raw):

    # Extract events using the mne.find_events function
    events = mne.find_events(raw, shortest_event=1)

    # Create a DataFrame for the events
    events_df = pd.DataFrame(events, columns=['onset', 'duration', 'value'])

    # The onset is in samples, convert to seconds by dividing by the sampling frequency
    events_df['onset'] = events_df['onset'] / raw.info['sfreq']

    # Set duration to 0 as it is typically not defined in event files unless specified
    events_df['duration'] = 0

    # Rename columns to match BIDS specification
    events_df.rename(columns={'onset': 'onset', 'duration': 'duration', 'value': 'trial_type'}, inplace=True)

    return events_df

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def plot_and_save_nii(nii_file, output_file):
    # Load the NIfTI file using nibabel
    img = nib.load(nii_file)
    data = img.get_fdata()
    
    # Get the middle slice for each axis (x, y, z)
    slice_x = data[data.shape[0] // 2, :, :]  # Sagittal slice
    slice_y = data[:, data.shape[1] // 2, :]  # Coronal slice
    slice_z = data[:, :, data.shape[2] // 2]  # Axial slice

    # Rotate the slices as specified
    slice_x = np.rot90(slice_x, k=3)  # 90 degrees clockwise
    slice_z = np.rot90(slice_z, k=2)  # 180 degrees

    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot each slice
    axes[0].imshow(slice_x.T, cmap='gray', origin='lower')
    axes[0].set_title('Sagittal Slice')
    axes[0].axis('off')

    axes[1].imshow(slice_y.T, cmap='gray', origin='lower')
    axes[1].set_title('Coronal Slice')
    axes[1].axis('off')

    axes[2].imshow(slice_z.T, cmap='gray', origin='lower')
    axes[2].set_title('Axial Slice')
    axes[2].axis('off')

    # Save the plot to a file
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def convert_mri_to_bids(participant, new_id, input_base_path, output_base_path):
    # Define paths
    mri_input_dir = os.path.join(input_base_path, participant, 'mri/deface')
    output_participant_dir = os.path.join(output_base_path, f'sub-{new_id}')
    anat_dir = os.path.join(output_participant_dir, 'anat')
    
    # Ensure the output directories exist
    os.makedirs(anat_dir, exist_ok=True)
    
    # Locate the defaced MRI file
    defaced_mri_file = os.path.join(mri_input_dir, '001_defaced.nii')
    if not os.path.isfile(defaced_mri_file):
        print(f"No defaced MRI file found for participant {participant} at {defaced_mri_file}.")
        return

    # Define the output file paths
    bids_mri_file = os.path.join(anat_dir, f'sub-{new_id}_T1w.nii.gz')
    plot_file = os.path.join(anat_dir, f'sub-{new_id}_T1w_deface_check_plot.png')
    json_file = os.path.join(anat_dir, f'sub-{new_id}_T1w.json')

    # Visualize and save the defaced MRI image
    plot_and_save_nii(defaced_mri_file, plot_file)
    
    # Copy the defaced MRI file to the BIDS output path
    shutil.copyfile(defaced_mri_file, bids_mri_file)
    print(f"Copied {defaced_mri_file} to {bids_mri_file}")

    # Extract and create JSON metadata file
    mri_metadata = extract_mri_metadata(defaced_mri_file)
    with open(json_file, 'w') as jf:
        json.dump(mri_metadata, jf, indent=4)
    print(f"Metadata JSON file created at {json_file}")

def extract_mri_metadata(nii_file):
    img = nib.load(nii_file)
    hdr = img.header

    # Extract metadata from the NIfTI header
    metadata = {"Manufacturer":"Siemens",
                "ManufacturersModelName":"3T Tim Trio",
                "MagneticFieldStrength":3,
                "ReceiveCoilName":"32Ch_Head",
                "ScanningSequence":"CBU_MPRAGE",
                "EchoTime":0.00299,
                "InversionTime":0.9,
                "FlipAngle":9,
                "RepetitionTime":2.250}
    
    # # Handle missing data gracefully
    # for key, value in metadata.items():
    #     if np.isnan(value):
    #         metadata[key] = "n/a"
    
    return metadata