import numpy as np
import os
from kymata.io.nkg import load_expression_set

# Define the base path for the expression sets and the output directory
# base_path = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ru_narr_en_native/language_pilots_all/expression_set"
# output_dir = "/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/ru_narr_en_native/language_pilots_all/fake_log"
base_path = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all'
output_dir = '/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all directories and files in the base path
for root, dirs, files in os.walk(base_path):
    for file in files:
        # if file.endswith(".nkg") and not os.path.isfile(os.path.join(output_dir, f"{os.path.splitext(file)[0]}_results.txt")):  # Assuming .nkg is the file extension for the expression sets
        if file.endswith(".nkg"):
            file_path = os.path.join(root, file)
            expression_data = load_expression_set(file_path)
            data_array = expression_data.scalp

            # Get the data and coordinates
            data = data_array.data  # Assuming this is a COO sparse array
            latency_coords = data_array.coords['latency'].values
            sensor_coords = data_array.coords['sensor'].values
            function_coords = data_array.coords['function'].values

            # Prepare the output file path
            output_file = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_results.txt")
            
            # Open the output file for writing
            with open(output_file, 'w') as out_f:
                # Iterate over each function (layer)
                for i, func_name in enumerate(function_coords):
                    # Extract data for the current function across all sensors and latencies
                    function_data = data[:, :, i].data

                    # Find the index of the maximum -log(pval)
                    sensor_ind = np.argmin(function_data, axis=None)
                    latency_ind = data[sensor_ind, :, i].coords[0][0]

                    peak_log_pval = -function_data[sensor_ind]
                    peak_lat = latency_coords[latency_ind]*1000

                    # Write the result to the file
                    out_f.write(f"{func_name}: peak lat: {peak_lat},   [sensor] ind: {sensor_ind},   -log(pval): {peak_log_pval}\n")