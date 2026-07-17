from logging import basicConfig, INFO
from pathlib import Path

from cyclopts import App

from kymata.io.config import load_config, get_root_dir
from kymata.io.logging import log_message, date_format
from kymata.preproc.data_cleansing import estimate_noise_cov


_app = App()


@_app.default
def main(
        config: str = "dataset4.yaml",
):
    """
    Create Trialwise Data
    
    Args:
        config (str): Name of the config .yaml file to use. 
    """
    config = load_config(Path(__file__).parent.parent / "dataset_config" / config)

    estimate_noise_cov(
        data_root_dir=get_root_dir(config),
        empty_room_estimate_year=str(config["meg_sss_noise_estimate_year"]),
        list_of_participants=config["participants"],
        dataset_directory_name=config["dataset_directory_name"],
        n_runs=config["number_of_runs"],
        cov_method=config["cov_method"],
        duration_emp=config["duration"],
        reg_method=config["reg_method"],
    )


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    _app()
