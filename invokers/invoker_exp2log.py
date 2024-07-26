import xarray as xr
import numpy as np
from kymata.io.nkg import load_expression_set


path = ''
expression_data = load_expression_set(path)

# Assuming 'expression_data' is your dataset and 'scalp' is the DataArray
data_array = expression_data.scalp

# Get the data and coordinates
data = data_array.data  # Assuming this is a COO sparse array
latency_coords = data_array.coords['latency'].values
sensor_coords = data_array.coords['sensor'].values
function_coords = data_array.coords['function'].values

# Ensure data is converted to dense if it's sparse
if hasattr(data, 'todense'):
    data = data.todense()

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

    print(f"{func_name}: peak lat: {peak_lat},   [sensor] ind: {sensor_ind},   -log(pval): {peak_log_pval}")