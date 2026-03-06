from logging import basicConfig, INFO, getLogger
from os import walk
from pathlib import Path

import numpy as np

from kymata.entities.expression import SensorExpressionSet
from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set


_logger = getLogger(__file__)


def main(input_dir: Path, output_dir: Path):
    # Iterate over all directories and files in the base path
    for root, dirs, files in walk(input_dir):
        for filename in files:
            if filename.endswith(".nkg"):
                file_path = Path(root) / filename
                main_single(file_path, output_dir)


def main_single(input_file: Path, output_dir: Path):
    _logger.info(f"Creating dummy log files from {input_file.name}")
    output_dir.mkdir(exist_ok=True, parents=False)

    expression_data = load_expression_set(input_file)
    if not isinstance(expression_data, SensorExpressionSet):
        raise NotImplementedError("Only works for sensor data")

    data_array = expression_data.scalp

    # Get the data and coordinates
    data = data_array.data  # Assuming this is a COO sparse array
    latency_coords = data_array.coords['latency'].values
    sensor_coords = data_array.coords['sensor'].values
    function_coords = data_array.coords['transform'].values

    # Prepare the output file path
    output_file = output_dir / f"{input_file.stem}_results.txt"

    # Open the output file for writing
    with output_file.open("w") as out_f:
        # Iterate over each function (layer)
        for i, func_name in enumerate(function_coords):
            # Extract data for the current function across all sensors and latencies
            function_data = data[:, :, i].data

            # Find the index of the maximum -log(pval)
            sensor_ind = np.argmin(function_data, axis=None)
            latency_ind = data[sensor_ind, :, i].coords[0][0]

            peak_log_pval = -function_data[sensor_ind]
            peak_lat = latency_coords[latency_ind] * 1000

            # Write the result to the file
            out_f.write(
                f"{func_name}: peak lat: {peak_lat},   [sensor] ind: {sensor_ind},   -log(pval): {peak_log_pval}\n")


if __name__ == '__main__':
    import argparse

    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, help="Input file or directory")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--multi", action="store_true", help="Run on all files in a directory")
    args = parser.parse_args()

    if args.multi:
        main(input_dir=args.input, output_dir=args.output)
    else:
        main_single(input_file=args.input, output_dir=args.output)
