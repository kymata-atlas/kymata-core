from logging import basicConfig, INFO
from pathlib import Path

from cyclopts import App

from kymata.io.config import load_config, get_root_dir
from kymata.io.logging import log_message, date_format
from kymata.preproc.hexel_current_estimation import (
    create_current_estimation_prerequisites, create_forward_model_and_inverse_solution, create_hexel_morph_maps,
    confirm_digitisation_locations)


_app = App()


@_app.default
def main(
        config: str = "dataset4.yaml",
):
    """
    Run hexel current extimation.

    Args:
        config (str): Path to the appropriate dataset config .yaml file.
    """
    config = load_config(Path(__file__).parent.parent / "dataset_config" / config)

    data_root_dir = get_root_dir(config)

    create_current_estimation_prerequisites(data_root_dir, config=config)

    create_forward_model_and_inverse_solution(data_root_dir, config=config)

    confirm_digitisation_locations(data_root_dir, config=config)

    create_hexel_morph_maps(data_root_dir, config=config)


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)

    _app()
