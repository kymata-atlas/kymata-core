from logging import basicConfig, INFO, getLogger
from pathlib import Path

from kymata.entities.expression import SensorExpressionSet
from kymata.io.logging import log_message, date_format
from kymata.io.nkg import load_expression_set, save_expression_set


_logger = getLogger(__file__)


def main(input_file: Path, output_dir: Path):
    _logger.info(f"Loading {input_file.name}")
    es = load_expression_set(input_file)
    if not isinstance(es, SensorExpressionSet):
        raise ValueError("Can only split sensor types for sensor expression sets")

    meg_sensors, eeg_sensors = split_meg_eeg_sensors(es)

    es_meg = es.subset_sensors(meg_sensors)
    es_eeg = es.subset_sensors(eeg_sensors)

    _logger.info(f"Saving MEG and EEG expression sets separately")
    save_expression_set(es_meg, output_dir / f"{input_file.stem}_meg{input_file.suffix}", overwrite=False)
    save_expression_set(es_eeg, output_dir / f"{input_file.stem}_eeg{input_file.suffix}", overwrite=False)


def split_meg_eeg_sensors(es: SensorExpressionSet) -> tuple[list[str], list[str]]:
    meg_sensors = [s for s in es.sensors if "MEG" in s]
    eeg_sensors = [s for s in es.sensors if "EEG" in s]
    return meg_sensors, eeg_sensors


if __name__ == '__main__':
    import argparse

    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", type=Path, help="Input .nkg file")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    args = parser.parse_args()

    main(input_file=args.input, output_dir=args.output)
