import os
import shutil
import yaml

# Path to the YAML config file
yaml_path = r'C:\Users\cy02\Documents\GitHub\camcan\kymata-core\dataset_config\camcan.yaml'

# Load participant IDs from YAML
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

participants = config.get('participants', [])

for participant in participants:
    src = fr'\\cbsu\data\Group\Camcan\ccrescan\meg\pipeline\release001\BIDSsep\noise\{participant}\meg\{participant}_task-noise_meg.fif'
    dest_dir = fr'\\cbsu\data\Imaging\projects\cbu\kymata\data\open-source\Camcan_movie\raw_emeg\{participant}\meg'
    dest = os.path.join(dest_dir, f'{participant}_task-noise_meg.fif')

    os.makedirs(dest_dir, exist_ok=True)
    try:
        shutil.copy2(src, dest)
        print(f'Copied {src} to {dest}')
    except Exception as e:
        print(f'Failed to copy for {participant}: {e}')