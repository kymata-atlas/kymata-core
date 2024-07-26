import xarray as xr
import numpy as np
import os
from kymata.io.nkg import load_expression_set

# Define the base path for the expression sets and the output directory
base_path = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/fc2_test/mfa/expression_set"
output_dir = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/fc2_test/fake_log"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all directories and files in the base path
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".nkg"):  # Assuming .nkg is the file extension for the expression sets
            file_path = os.path.join(root, file)
            expression_data = load_expression_set(file_path)
            data_array = expression_data.scalp

            # Get the data and coordinates
            data = data_array.data  # Assuming this is a COO sparse array
            latency_coords = data_array.coords['latency'].values
            sensor_coords = data_array.coords['sensor'].values
            function_coords = data_array.coords['function'].values

            # Ensure data is converted to dense if it's sparse
            if hasattr(data, 'todense'):
                data = data.todense()

            # Prepare the output file path
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_results.txt")
            
            # Open the output file for writing
            with open(output_file, 'w') as out_f:
                # Iterate over each function (layer)
                for i, func_name in enumerate(function_coords):
                    # Extract data for the current function across all sensors and latencies
                    function_data = data[:, :, i]

                    # Find the index of the maximum -log(pval)
                    max_index = np.unravel_index(np.argmax(function_data, axis=None), function_data.shape)

                    sensor_ind = max_index[0]
                    latency_ind = max_index[1]

                    peak_log_pval = function_data[sensor_ind, latency_ind]
                    peak_lat = latency_coords[latency_ind]

                    # Write the result to the file
                    out_f.write(f"{func_name}: peak lat: {peak_lat},   [sensor] ind: {sensor_ind},   -log(pval): {peak_log_pval}\n")