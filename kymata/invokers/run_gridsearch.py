from logging import getLogger, basicConfig, INFO
from pathlib import Path
import argparse
import time
from sys import stdout

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.transform import load_transform
from kymata.io.config import load_config
from kymata.io.logging import log_message, date_format
from kymata.io.nkg import save_expression_set
from kymata.io.layouts import SensorLayout, MEGLayout, EEGLayout
from kymata.preproc.source import load_emeg_pack
from kymata.plot.expression import expression_plot


_default_output_dir = Path(data_root_path(), "output")
_logger = getLogger(__file__)


def get_config_value_with_fallback(config: dict, config_key: str, fallback):
    """
    Get a value from the config, with a default fallback, and a notification explaining this.
    """
    try:
        return config[config_key]
    except KeyError:
        _logger.error(f'Config did not contain any value for "{config_key}", falling back to default value {fallback}')
        return fallback


def main():
    _default_output_dir.mkdir(exist_ok=True, parents=False)

    parser = argparse.ArgumentParser(description="Gridsearch Params")

    # Dataset specific
    parser.add_argument("--config", type=str, required=True,
                        help="Either the path to the config file to be used, or the name of the config file to be used "
                             "if included with kymata-core.")

    parser.add_argument("--emeg-dir", type=str,
                        default="interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/",
                        help="EMEG directory, relative to base dir")

    # Analysis specific
    parser.add_argument("--overwrite", action="store_true", help="Silently overwrite existing files.")

    # Participants
    parser.add_argument("--single-participant-override", type=str, default=None, required=False,
                        help="Supply to run only on one participant")
    parser.add_argument("--ave-mode", type=str, default="ave", choices=["ave", "concatenate"],
                        help="`ave`: average over the list of repetitions. `concatenate`: treat them as extra data.")

    # Transforms
    parser.add_argument("--input-stream", type=str, required=True, choices=["auditory", "visual", "tactile"],
                        help="The input stream for the transforms being tested.")
    parser.add_argument("--transform-path", type=str, default="predicted_function_contours/GMSloudness/stimulisig",
                        help="Location of transform stimulisig. Either supply relative to the data dir, or as an "
                             "absolute path. Both `.npz` and `.mat` extensions will be checked, in that order.")
    parser.add_argument("--transform-name", type=str, nargs="+", help="transform names in stimulisig")
    parser.add_argument("--replace-nans", type=str, required=False, choices=["zero", "mean"], default=None,
                        help="If the transform contour contains NaN values, "
                             "this will replace them with the specified values.")
    parser.add_argument("--transform-sample-rate", type=float, required=False, default=1000,
                        help="The sample rate of the transform contour.")

    # For source space
    parser.add_argument("--use-inverse-operator", action="store_true",
                        help="Use inverse operator to conduct gridsearch in source space.")
    parser.add_argument("--morph", action="store_true",
                        help="Morph hexel data to fs-average space prior to running gridsearch. "
                             "Only has an effect if an inverse operator is specified.",)
    parser.add_argument("--inverse-operator-suffix", type=str, default="_ico5-3L-loose02-cps-nodepth-fusion-inv.fif",
                        help="inverse solution suffix")

    parser.add_argument("--snr", type=float, default=3, help="Inverse solution SNR")
    parser.add_argument("--resample", type=float, required=False, default=200, help="Resample rate in Hz.")

    # General gridsearch
    parser.add_argument("--seconds-per-split", type=float, default=1,
                        help="Seconds in each split of the recording, also maximum range of latencies being checked")
    parser.add_argument("--n-splits", type=int, default=400,
                        help="Number of splits to split the recording into, "
                             "(set to stimulus_length/seconds_per_split for full file)")
    parser.add_argument("--n-derangements", type=int, default=6,
                        help="Number of deragements for the null distribution")
    parser.add_argument("--start-latency", type=float, default=-0.2,
                        help="Earliest latency to check in cross correlation")
    parser.add_argument("--emeg-t-start", type=float, default=-0.2,
                        help="Start of the emeg evoked files relative to the start of the transform")

    # Output paths
    parser.add_argument("--save-name", type=str, required=False, help="Specify the name of the saved .nkg file.")
    parser.add_argument("--save-expression-set-location", type=Path, default=Path(_default_output_dir),
                        help="Save the results of the gridsearch into an ExpressionSet .nkg file")
    parser.add_argument("--save-plot-location", type=Path, default=Path(_default_output_dir),
                        help="Save an expression plots, and other plots, in this location")
    parser.add_argument("--plot-top-channels", action="store_true",
                        help="Plots the p-values and correlations of the top channels in the gridsearch.")

    args = parser.parse_args()

    specified_config_file = Path(args.config)
    if specified_config_file.exists():
        _logger.info(f"Loading config file from {str(specified_config_file)}")
        dataset_config = load_config(str(specified_config_file))
    else:
        default_config_file = Path(Path(__file__).parent.parent.parent, "dataset_config", args.config)
        _logger.info(f"Config specified by name. Loading config file from {str(default_config_file)}")
        dataset_config = load_config(str(default_config_file))

    # Config defaults
    participants = dataset_config.get("participants")
    base_dir = Path(
        "/imaging/projects/cbu/kymata/data/",
        dataset_config.get("dataset_directory_name", "dataset_4-english-narratives"),
    )
    inverse_operator_dir = dataset_config.get("inverse_operator")

    input_stream = args.input_stream
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
    
    if input_stream != "tactile":

        reps = [f"_rep{i}" for i in range(8)] + [
            "-ave"
        ]  # most of the time we will only use the -ave, not the individual reps
        if args.single_participant_override is not None:
            if args.ave_mode == "ave":
                emeg_filenames = [args.single_participant_override + "-ave"]
            elif args.ave_mode == "concatenate":
                print("Concatenating repetitions together")
                emeg_filenames = [args.single_participant_override + r for r in reps[:-1]]
        else:
            emeg_filenames = [p + "-ave" for p in participants]

        start = time.time()

        if (
            (len(emeg_filenames) > 1)
            and (not args.morph)
            and (args.ave_mode == "ave")
            and args.use_inverse_operator
        ):
            raise ValueError(
                "Averaging source-space results without morphing to a common space. "
                + "If you are averaging over multiple participants you must morph to a common space."
            )

        # Load data
        emeg_path = Path(base_dir, args.emeg_dir)
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

        channel_space = "source" if args.use_inverse_operator else "sensor"

        _logger.info("Starting Kymata Gridsearch")
        _logger.info(f"Dataset: {dataset_config.get('dataset_directory_name')}")
        _logger.info(f"Transforms to be tested: {args.transform_name}")
        _logger.info(f"Gridsearch will be applied in {channel_space} space")
        if args.use_inverse_operator:
            _logger.info(f"Inverse operator: {args.inverse_operator_suffix}")
        if args.morph:
            _logger.info("Morphing to common space")

        t0 = time.time()

        emeg_values, ch_names, n_reps = load_emeg_pack(
            emeg_filenames,
            emeg_dir=emeg_path,
            morph_dir=morph_dir if args.morph else None,
            need_names=True,
            ave_mode=args.ave_mode,
            inverse_operator_dir=inverse_operator_dir
            if args.use_inverse_operator
            else None,
            inverse_operator_suffix=args.inverse_operator_suffix,
            snr=args.snr,
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
        if Path(args.transform_path).exists():
            transform_path = Path(args.transform_path)
        else:
            transform_path = Path(base_dir, args.transform_path)
        _logger.info(f"Loading transforms from {str(transform_path)}")

        emeg_sample_rate = float(dataset_config.get("sample_rate", 1000))

        sensor_layout = SensorLayout(
            meg=MEGLayout(dataset_config["meg_sensor_layout"]),
            eeg=EEGLayout(dataset_config["eeg_sensor_layout"]),
        )

        for transform_name in args.transform_name:
            _logger.info(f"Running gridsearch on {transform_name}")
            transform = load_transform(
                transform_path,
                trans_name=transform_name,
                replace_nans=args.replace_nans,
                bruce_neurons=(5, 10),
                sample_rate=args.transform_sample_rate,
            )

            # Resample transform to match target sample rate if specified, else emeg sample rate
            transform_resample_rate = args.resample if args.resample is not None else emeg_sample_rate
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
                seconds_per_split=args.seconds_per_split,
                n_derangements=args.n_derangements,
                n_splits=args.n_splits,
                n_reps=n_reps,
                emeg_sample_rate=emeg_sample_rate,
                start_latency=args.start_latency,
                plot_location=args.save_plot_location,
                emeg_t_start=args.emeg_t_start,
                stimulus_shift_correction=stimulus_shift_correction,
                stimulus_delivery_latency=stimulus_delivery_latency,
                plot_top_five_channels=args.plot_top_channels,
                overwrite=args.overwrite,
                emeg_layout=sensor_layout,
            )

            if combined_expression_set is None:
                combined_expression_set = es
            else:
                combined_expression_set += es

        assert combined_expression_set is not None

        combined_names: str
        if args.save_name is not None and len(args.save_name) > 0:
            combined_names = args.save_name
        elif len(args.transform_name) > 2:
            combined_names = f"{len(args.transform_name)}_transforms_gridsearch"
        else:
            combined_names = "_+_".join(args.transform_name) + "_gridsearch"

        if args.save_expression_set_location is not None:
            es_save_path = Path(
                args.save_expression_set_location, combined_names
            ).with_suffix(".nkg")
            _logger.info(f"Saving expression set to {es_save_path!s}")
            save_expression_set(
                combined_expression_set,
                to_path_or_file=es_save_path,
                overwrite=args.overwrite,
            )

        if args.single_participant_override is not None:
            fig_save_path = Path(
                args.save_plot_location,
                combined_names + f"_{args.single_participant_override}",
            ).with_suffix(".png")
        else:
            fig_save_path = Path(args.save_plot_location, combined_names).with_suffix(".png")
        _logger.info(f"Saving expression plot to {fig_save_path!s}")
        expression_plot(
            combined_expression_set,
            paired_axes=channel_space == "source",
            save_to=fig_save_path,
            overwrite=args.overwrite,
        )

        total_time_in_seconds = time.time() - start
        _logger.info(
            f'Time taken for code to run: {time.strftime("%H:%M:%S", time.gmtime(total_time_in_seconds))} ({total_time_in_seconds:.4f}s)'
        )

    else:

        all_es = None

        for i in range(1, 6):
            
            emeg_filenames = [p + f"_run{i}" for p in participants]

            start = time.time()

            # Load data
            emeg_path = Path(base_dir, args.emeg_dir)
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

            channel_space = "source" if args.use_inverse_operator else "sensor"

            _logger.info("Starting Kymata Gridsearch")
            _logger.info(f"Dataset: {dataset_config.get('dataset_directory_name')}")
            _logger.info(f"Transforms to be tested: {args.transform_name}")
            _logger.info(f"Gridsearch will be applied in {channel_space} space")
            if args.use_inverse_operator:
                _logger.info(f"Inverse operator: {args.inverse_operator_suffix}")
            if args.morph:
                _logger.info("Morphing to common space")

            t0 = time.time()

            emeg_values, ch_names, n_reps = load_emeg_pack(
                emeg_filenames,
                emeg_dir=emeg_path,
                morph_dir=morph_dir if args.morph else None,
                need_names=True,
                ave_mode=args.ave_mode,
                inverse_operator_dir=inverse_operator_dir
                if args.use_inverse_operator
                else None,
                inverse_operator_suffix=args.inverse_operator_suffix,
                snr=args.snr,
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
            if Path(args.transform_path).exists():
                transform_path = Path(args.transform_path)
            else:
                transform_path = Path(base_dir, args.transform_path)
            _logger.info(f"Loading transforms from {str(transform_path)}")

            emeg_sample_rate = float(dataset_config.get("sample_rate", 1000))

            sensor_layout = SensorLayout(
                meg=MEGLayout(dataset_config["meg_sensor_layout"]),
                eeg=EEGLayout(dataset_config["eeg_sensor_layout"]),
            )

            for transform_name in args.transform_name:
                _logger.info(f"Running gridsearch on {transform_name}")
                transform = load_transform(
                    transform_path,
                    trans_name=transform_name + f"_{i}",
                    replace_nans=args.replace_nans,
                    bruce_neurons=(5, 10),
                    sample_rate=args.transform_sample_rate,
                )

                # Resample transform to match target sample rate if specified, else emeg sample rate
                transform_resample_rate = args.resample if args.resample is not None else emeg_sample_rate
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
                    seconds_per_split=args.seconds_per_split,
                    n_derangements=args.n_derangements,
                    n_splits=args.n_splits,
                    n_reps=n_reps,
                    emeg_sample_rate=emeg_sample_rate,
                    start_latency=args.start_latency,
                    plot_location=args.save_plot_location,
                    emeg_t_start=args.emeg_t_start,
                    stimulus_shift_correction=stimulus_shift_correction,
                    stimulus_delivery_latency=stimulus_delivery_latency,
                    plot_top_five_channels=args.plot_top_channels,
                    overwrite=args.overwrite,
                    emeg_layout=sensor_layout,
                )

                if combined_expression_set is None:
                    combined_expression_set = es
                else:
                    combined_expression_set += es

            assert combined_expression_set is not None

            if all_es is None:
                all_es = combined_expression_set
            else:
                all_es += combined_expression_set

        es_save_path = Path(
                args.save_expression_set_location, 'all_tactile'
            ).with_suffix(".nkg")
        fig_save_path = Path(args.save_plot_location, 'all_tactile').with_suffix(".png")


        save_expression_set(
                    all_es,
                    to_path_or_file=es_save_path,
                    overwrite=args.overwrite,
                )
        expression_plot(
            all_es,
            paired_axes=channel_space == "source",
            save_to=fig_save_path,
            overwrite=args.overwrite,
        )


        total_time_in_seconds = time.time() - start
        _logger.info(
            f'Time taken for code to run: {time.strftime("%H:%M:%S", time.gmtime(total_time_in_seconds))} ({total_time_in_seconds:.4f}s)'
        )      


if __name__ == "__main__":
    basicConfig(format=log_message, datefmt=date_format, level=INFO)
    main()
