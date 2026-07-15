from logging import basicConfig, INFO
from pathlib import Path

from cyclopts import App

from kymata.entities.rudimentary import get_coerce
from kymata.io.config import load_config, get_root_dir
from kymata.io.logging import log_message, date_format
from kymata.preproc.data_cleansing import create_trialwise_data


_app = App()


@_app.default
def main(
        config: str = "dataset4.yaml",
) -> None:
    """
    Create trialwise data

    Args:
        config:

    """
    config = load_config(Path(__file__).parent.parent / "dataset_config" / config)
    check_drift = config.get("check_drift_and_delay", False)

    create_trialwise_data(
        data_root_dir=get_root_dir(config),
        dataset_directory_name=config["dataset_directory_name"],
        list_of_participants=config["participants"],
        repetitions_per_runs=config["repetitions_per_runs"],
        data_sample_rate=config["sample_rate"],
        stimulus_length=config["stimulus_length"],
        number_of_runs=config["number_of_runs"],
        latency_range=config["latency_range"],
        check_drift_with_audio_stim=config["audio_stimulus_file"] if check_drift else None,
        reference_drift=get_coerce(config, key="audio_delivery_drift_correction", coerce=float, default=None),
        reference_delay=get_coerce(config, key="audio_delivery_latency", coerce=float, default=None),
    )


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    _app()
