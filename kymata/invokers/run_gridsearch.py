from logging import getLogger, basicConfig, INFO
from pathlib import Path
import time
from sys import stdout
from typing import Optional, Literal, Annotated

from cyclopts import App, Parameter

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.transform import load_transform
from kymata.io.config import load_config
from kymata.io.logging import log_message, date_format
from kymata.io.nkg import save_expression_set
from kymata.io.layouts import SensorLayout, MEGLayout, EEGLayout
from kymata.preproc.source import load_emeg_pack
from kymata.plot.expression import expression_plot
from kymata.system.reflection import kymata_installed_as_dependency


_default_output_dir = Path(data_root_path(), "output")
_logger = getLogger(__file__)
_app = App()


def get_config_value_with_fallback(config: dict, config_key: str, fallback):
    """
    Get a value from the config, with a default fallback, and a notification explaining this.
    """
    try:
        return config[config_key]
    except KeyError:
        _logger.error(f'Config did not contain any value for "{config_key}", falling back to default value {fallback}')
        return fallback


@_app.default
def main(
        config: str,
        # Transforms
        transform_name: Annotated[list[str], Parameter(consume_multiple=True)],
        input_stream: Literal["auditory", "visual", "tactile"],
        transform_path: str = "predicted_function_contours/GMSloudness/stimulisig",
        transform_sample_rate: float = 1000,
        replace_nans: Optional[Literal["zero", "mean"]] = None,
        # Paths
        emeg_dir: str = "interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/",
        data_root: Path = Path("/imaging/projects/cbu/kymata/data/"),
        save_name: Optional[str] = None,
        save_expression_set_location: Optional[Path] = None,
        save_plot_location: Optional[Path] = None,
        # Participants
        single_participant_override: Optional[str] = None,
        ave_mode: Literal["ave", "concatenate"] = "ave",
        # Data
        resample: float = 200,
        seconds_per_split: float = 1,
        n_splits: int = 400,
        n_derangements: int = 6,
        start_latency: float = -0.2,
        emeg_t_start: float = -0.2,
        # For source space
        use_inverse_operator: bool = False,
        inverse_operator_suffix: str = "_ico5-3L-loose02-cps-nodepth-fusion-inv.fif",
        morph: bool = False,
        snr: float = 3,
        # Output options
        plot_top_channels: bool = False,
        # Analysis specific
        overwrite: bool = False,
) -> None:
    """
    Run the plain gridsearch.
    
    Args:
        config: Either the path to the config file to be used, or the name of the config file to be used if included with kymata-core.
        transform_name: "transform names in stimulisig".
        input_stream: The input stream for the transforms being tested.
        transform_path: Location of transform stimulisig. Either supply relative to the data dir, or as an absolute path. Both `.npz` and `.mat` extensions will be checked, in that order.
        transform_sample_rate: The original sample rate of the transform contour.
        replace_nans: If the transform contour contains NaN values, this will replace them with the specified values.
        emeg_dir: EMEG directory, relative to base dir
        data_root: Root directory of kymata data
        save_name: Specify the name of the saved .nkg file.
        save_expression_set_location: Save the results of the gridsearch into an ExpressionSet .nkg file.
        save_plot_location: Save an expression plots, and other plots, in this location.
        single_participant_override: Supply to run only on one participant
        ave_mode: `ave`: average over the list of repetitions. `concatenate`: treat them as extra data.
        resample: Resample rate for both transform and EMEG data, in Hz. (E.g. if the transform sample rate is 1000Hz, this can be 100, 200, 250, 500, 1000.
        seconds_per_split: Seconds in each split of the recording, also maximum range of latencies being checked
        n_splits: Number of splits to split the recording into, (set to stimulus_length/seconds_per_split for full file).
        n_derangements: Number of deragements for the null distribution.
        start_latency: Earliest latency to check in cross correlation.
        emeg_t_start: Start of the emeg evoked files relative to the start of the transform.
        use_inverse_operator: Use inverse operator to conduct gridsearch in source space.
        inverse_operator_suffix: Inverse solution suffix.
        morph: Morph hexel data to fs-average space prior to running gridsearch. Only has an effect if an inverse operator is specified.
        snr: Inverse solution SNR.
        plot_top_channels: Plots the p-values and correlations of the top channels in the gridsearch.
        overwrite: Silently overwrite existing files.
    """

    # Save locations are non-optional when running as a dependency
    if save_expression_set_location is None:
        if kymata_installed_as_dependency():
            raise ValueError("Must specify expression set save location when running kymata as a dependency")
        else:
            save_expression_set_location = _default_output_dir
    if save_plot_location is None:
        if kymata_installed_as_dependency():
            raise ValueError("Must specify plot save location when running kymata as a dependency")
        else:
            save_plot_location = _default_output_dir

    # Get config
    specified_config_file = Path(config)
    if specified_config_file.exists():
        _logger.info(f"Loading config file from {str(specified_config_file)}")
        dataset_config = load_config(str(specified_config_file))
    else:
        default_config_file = Path(Path(__file__).parent.parent, "dataset_config", config)
        _logger.info(f"Config specified by name. Loading config file from {str(default_config_file)}")
        dataset_config = load_config(str(default_config_file))

    # Config defaults
    participants = dataset_config.get("participants")
    base_dir = Path(
        data_root,
        dataset_config.get("dataset_directory_name", "dataset_4-english-narratives"),
    )
    inverse_operator_dir = dataset_config.get("inverse_operator")

    input_stream = input_stream
    if input_stream == "auditory":
        stimulus_shift_correction = dataset_config["audio_delivery_drift_correction"]
        stimulus_delivery_latency = dataset_config["audio_delivery_latency"]
    elif input_stream == "visual":
        stimulus_shift_correction = dataset_config["visual_delivery_drift_correction"]
        stimulus_delivery_latency = dataset_config["visual_delivery_latency"]
    elif input_stream == "tactile":
        stimulus_shift_correction = dataset_config["tactile_delivery_drift_correction"]
        stimulus_delivery_latency = dataset_config["tactile_delivery_latency"]
    else:
        raise NotImplementedError()

    reps = [f"_rep{i}" for i in range(8)] + [
        "-ave"
    ]  # most of the time we will only use the -ave, not the individual reps
    if single_participant_override is not None:
        if ave_mode == "ave":
            emeg_filenames = [single_participant_override + "-ave"]
        elif ave_mode == "concatenate":
            print("Concatenating repetitions together")
            emeg_filenames = [single_participant_override + r for r in reps[:-1]]
    else:
        emeg_filenames = [p + "-ave.fif" for p in participants]

    start = time.time()

    if (
        (len(emeg_filenames) > 1)
        and (not morph)
        and (ave_mode == "ave")
        and use_inverse_operator
    ):
        raise ValueError(
            "Averaging source-space results without morphing to a common space. "
            + "If you are averaging over multiple participants you must morph to a common space."
        )

    # Load data
    emeg_path = Path(base_dir, emeg_dir)
    morph_dir = Path(
        base_dir,
        "interim_preprocessing_files",
        "4_hexel_current_reconstruction",
        "morph_maps",
    )
    invsol_npy_dir = Path(
        base_dir,
        "interim_preprocessing_files",
        "4_hexel_current_reconstruction",
        "npy_invsol",
    )
    inverse_operator_dir = Path(base_dir, inverse_operator_dir)

    channel_space = "source" if use_inverse_operator else "sensor"

    _logger.info("Starting Kymata Gridsearch")
    _logger.info(f"Dataset: {dataset_config.get('dataset_directory_name')}")
    _logger.info(f"Transforms to be tested: {transform_name}")
    _logger.info(f"Gridsearch will be applied in {channel_space} space")
    if use_inverse_operator:
        _logger.info(f"Inverse operator: {inverse_operator_suffix}")
    if morph:
        _logger.info("Morphing to common space")

    t0 = time.time()

    emeg_values, ch_names, n_reps = load_emeg_pack(
        emeg_filenames,
        emeg_dir=emeg_path,
        morph_dir=morph_dir if morph else None,
        ave_mode=ave_mode,
        inverse_operator_dir=inverse_operator_dir
        if use_inverse_operator
        else None,
        inverse_operator_suffix=inverse_operator_suffix,
        snr=snr,
        old_morph=False,
        invsol_npy_dir=invsol_npy_dir,
        ch_names_path=Path(invsol_npy_dir, "ch_names.npy"),
    )

    time_to_load = time.time() - t0
    print(f"Time to load emeg: {time_to_load:.4f}")
    stdout.flush()  # make sure the above print statement shows up as soon as print is called
    _logger.info(f"Time to load emeg: {time_to_load:.4f}")

    combined_expression_set = None

    # Get stimulisig path
    if Path(transform_path).exists():
        transform_path = Path(transform_path)
    else:
        transform_path = Path(base_dir, transform_path)
    _logger.info(f"Loading transforms from {str(transform_path)}")

    emeg_sample_rate = float(dataset_config.get("sample_rate", 1000))

    sensor_layout = SensorLayout(
        meg=MEGLayout(dataset_config["meg_sensor_layout"]),
        eeg=EEGLayout(dataset_config["eeg_sensor_layout"]),
    )

    for transform_name in transform_name:
        _logger.info(f"Running gridsearch on {transform_name}")
        transform = load_transform(
            transform_path,
            trans_name=transform_name,
            replace_nans=replace_nans,
            bruce_neurons=(5, 10),
            sample_rate=transform_sample_rate,
        )

        # Resample transform to match target sample rate if specified, else emeg sample rate
        transform_resample_rate = resample if resample is not None else emeg_sample_rate
        if transform.sample_rate != transform_resample_rate:
            _logger.info(f"Transform sample rate ({transform.sample_rate} Hz) doesn't match target sample rate "
                         f"({transform_resample_rate} Hz). Transform will be resampled to match. "
                         f"({transform.sample_rate} → {transform_resample_rate} Hz)")
            transform = transform.resampled(transform_resample_rate)

        es = do_gridsearch(
            emeg_values=emeg_values,
            channel_names=ch_names,
            channel_space=channel_space,
            transform=transform,
            seconds_per_split=seconds_per_split,
            n_derangements=n_derangements,
            n_splits=n_splits,
            n_reps=n_reps,
            emeg_sample_rate=emeg_sample_rate,
            start_latency=start_latency,
            plot_location=save_plot_location,
            emeg_t_start=emeg_t_start,
            stimulus_shift_correction=stimulus_shift_correction,
            stimulus_delivery_latency=stimulus_delivery_latency,
            plot_top_five_channels=plot_top_channels,
            overwrite=overwrite,
            emeg_layout=sensor_layout,
            seed=dataset_config.get('random_seed_gridsearch', None),
        )

        if combined_expression_set is None:
            combined_expression_set = es
        else:
            combined_expression_set += es

    assert combined_expression_set is not None

    combined_names: str
    if save_name is not None and len(save_name) > 0:
        combined_names = save_name
    elif len(transform_name) > 2:
        combined_names = f"{len(transform_name)}_transforms_gridsearch"
    else:
        combined_names = "_+_".join(transform_name) + "_gridsearch"

    if save_expression_set_location is not None:
        es_save_path = Path(
            save_expression_set_location, combined_names
        ).with_suffix(".nkg")
        _logger.info(f"Saving expression set to {es_save_path!s}")
        save_expression_set(
            combined_expression_set,
            to_path_or_file=es_save_path,
            overwrite=overwrite,
        )

    if single_participant_override is not None:
        fig_save_path = Path(
            save_plot_location,
            combined_names + f"_{single_participant_override}",
        ).with_suffix(".png")
    else:
        fig_save_path = Path(save_plot_location, combined_names).with_suffix(".png")
    _logger.info(f"Saving expression plot to {fig_save_path!s}")
    expression_plot(
        combined_expression_set,
        color={
                "IL":  "#b11e34",
                "IL1": "#a201e9",
                "IL2": "#a201e9",
                "IL3": "#a201e9",
                "IL4": "#a201e9",
                "IL5": "#a201e9",
                "IL6": "#a201e9",
                "IL7": "#a201e9",
                "IL8": "#a201e9",
                "IL9": "#a201e9",
                "STL": "#d388b5",
            },
        paired_axes=channel_space == "source",
        save_to=fig_save_path,
        overwrite=overwrite,
    )

    total_time_in_seconds = time.time() - start
    _logger.info(
        f'Time taken for code to run: {time.strftime("%H:%M:%S", time.gmtime(total_time_in_seconds))} ({total_time_in_seconds:.4f}s)'
    )


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    _app()
