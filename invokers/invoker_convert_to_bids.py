import os
import json
from kymata.io.bids import convert_meg_to_bids, convert_mri_to_bids, participant_mapping
from tqdm import tqdm

# Input and output paths
input_base_path = '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/raw_emeg'  # Update to your base directory containing the participants
input_base_path_mri = '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives/raw_mri_structurals'  # Update to your base directory containing the participants
output_base_path = '/imaging/projects/cbu/kymata/data/for-export-as-BIDS'  # Path to BIDS dataset directory
task = 'en'

# # Create dataset description
# dataset_description = {
#     "Name": "Kymata-SOTO",
#     "BIDSVersion": "1.9.0",
#     "License": "CC0",
#     "Authors": ["Andrew Thwaites", "Chentianyi Yang"]
# }
# dataset_description_path = os.path.join(output_base_path, 'dataset_description.json')
# with open(dataset_description_path, 'w') as f:
#     json.dump(dataset_description, f, indent=4)

# # Create README file
# readme_content = "Kymata Soto Language Dataset: a high-quality electro- and magneto-encephalographic dataset for understanding speech and language processing"
# readme_path = os.path.join(output_base_path, 'README')
# with open(readme_path, 'w') as f:
#     f.write(readme_content)

# Process each participant
for participant in tqdm(os.listdir(input_base_path)):
    if participant in participant_mapping:
        new_id = participant_mapping[participant]
        # convert_meg_to_bids(participant, new_id, task, input_base_path, output_base_path)
        convert_mri_to_bids(participant, new_id, input_base_path_mri, output_base_path)
