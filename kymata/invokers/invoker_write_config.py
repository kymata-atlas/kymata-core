import os
import yaml

# Paths
# raw_emeg_dir = r'\\cbsu\data\Imaging\projects\cbu\kymata\data\open-source\Camcan_story\raw_emeg'
# config_path = r'C:\Users\cy02\Documents\GitHub\camcan\kymata-core\dataset_config\camcan_story.yaml'

raw_emeg_dir = r'/imaging/projects/cbu/kymata/data/open-source/Camcan_story/raw_emeg'
config_path = r'/imaging/projects/cbu/kymata/analyses/tianyi/camcan/kymata-core/dataset_config/camcan_story.yaml'

# Get folder names (participants)
participants = [
    name for name in os.listdir(raw_emeg_dir)
    if os.path.isdir(os.path.join(raw_emeg_dir, name))
]

# Load existing YAML or create new
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}

# Update participants field
config['participants'] = participants

# Write back to YAML using PyYAML

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
