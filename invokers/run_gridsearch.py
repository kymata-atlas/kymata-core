from pathlib import Path
import argparse
import time
import os

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.config import load_config
from kymata.preproc.source import load_emeg_pack
from kymata.io.nkg import save_expression_set
from kymata.plot.plot import expression_plot

_default_output_dir = Path(data_root_path(), "output")


def get_config_value_with_fallback(config: dict, config_key: str, fallback):
    """
    Get a value from the config, with a default fallback, and a notification explaining this.
    """
    try:
        return config[config_key]
    except KeyError:
        print(f"Config did not contain any value for \"{config_key}\", falling back to default value {fallback}")
        return fallback


def main():

    _default_output_dir.mkdir(exist_ok=True, parents=False)

    parser = argparse.ArgumentParser(description='Gridsearch Params')

    # Dataset specific
    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--emeg-dir', type=str, default='interim_preprocessing_files/3_trialwise_sensorspace/evoked_data/', help='emeg directory, relative to base dir')

    # Analysis specific
    parser.add_argument('--overwrite', action="store_true", help="Silently overwrite existing files.")

    # Participants
    parser.add_argument('--single-participant-override', type=str, default=None, required=False, help='Supply to run only on one participant')
    parser.add_argument('--ave-mode',                    type=str, default="ave", choices=["ave", "concatenate"], help='`ave`: average over the list of repetitions. `concatenate`: treat them as extra data.')

    # Functions
    parser.add_argument('--function-name', type=str, nargs="+", help='function names in stimulisig')
    parser.add_argument('--function-path', type=str, default='predicted_function_contours/GMSloudness/stimulisig', help='location of function stimulisig')
    parser.add_argument('--asr-option', type=str, default="ave",
                        help='Whether to get the output from all neurons (all) or do the average (ave)')
    parser.add_argument('--num-neurons', type=int, default=512,
                        help='Number of neurons in each layer')
    # For source space
    parser.add_argument('--use-inverse-operator',    action="store_true", help="Use inverse operator to conduct gridsearch in source space.")
    parser.add_argument('--morph',                   action="store_true", help="Morph hexel data to fs-average space prior to running gridsearch. Only has an effect if an inverse operator is specified.")
    parser.add_argument('--inverse-operator-suffix', type=str, default="_ico5-3L-loose02-cps-nodepth-fusion-inv.fif", help='inverse solution suffix')

    parser.add_argument('--snr',             type=float, default=3, help='inverse solution snr')
    parser.add_argument('--downsample-rate', type=int,   default=5, help='downsample_rate - DR=5 is equivalent to 200Hz, DR=2 => 500Hz, DR=1 => 1kHz')

    parser.add_argument('--seconds-per-split', type=float, default=1, help='seconds in each split of the recording, also maximum range of latencies being checked')
    parser.add_argument('--n-splits',          type=int, default=400, help='number of splits to split the recording into, (set to 400/seconds_per_split for full file)')
    parser.add_argument('--n-derangements',    type=int, default=5, help='number of deragements for the null distribution')
    parser.add_argument('--start-latency',     type=float, default=-200, help='earliest latency to check in cross correlation')
    parser.add_argument('--emeg-t-start',      type=float, default=-200, help='start of the emeg evoked files relative to the start of the function')

    # Output paths
    parser.add_argument('--save-name', type=str, required=False, help="Specify the name of the saved .nkg file.")
    parser.add_argument('--save-expression-set-location', type=Path, default=Path(_default_output_dir), help="Save the results of the gridsearch into an ExpressionSet .nkg file")
    parser.add_argument('--save-plot-location', type=Path, default=Path(_default_output_dir), help="Save an expression plots, and other plots, in this location")

    args = parser.parse_args()

    dataset_config = load_config(str(Path(Path(__file__).parent.parent, "dataset_config", args.config)))

    # Config defaults
    participants = dataset_config.get('participants')
    audio_shift_correction = get_config_value_with_fallback(dataset_config, "audio_delivery_shift_correction", fallback=0)
    base_dir = Path('/imaging/projects/cbu/kymata/data/', dataset_config.get('dataset_directory_name', 'dataset_4-english-narratives'))
    inverse_operator_dir = dataset_config.get('inverse_operator')

    os.makedirs(args.save_plot_location, exist_ok=True)
    os.makedirs(args.save_expression_set_location, exist_ok=True)

    reps = [f'_rep{i}' for i in range(8)] + ['-ave']
    if args.single_participant_override is not None:
        emeg_filenames = [args.single_participant_override + "-ave"]
    else:
        emeg_filenames = [
            p + r
            for p in participants
            for r in reps[-1:]
        ]

    start = time.time()

    if (len(emeg_filenames) > 1) and (not args.morph) and (args.ave_mode == "ave") and args.use_inverse_operator:
        raise ValueError(
            "Averaging source-space results without morphing to a common space. " +
            "If you are averaging over multiple participants you must morph to a common space.")

    # Load data
    emeg_path = Path(base_dir, args.emeg_dir)
    morph_dir = Path(base_dir, "interim_preprocessing_files", "4_hexel_current_reconstruction", "morph_maps")
    inverse_operator_dir = Path(base_dir, inverse_operator_dir)

    channel_space = "source" if args.use_inverse_operator else "sensor"

    print(f"Gridsearch in {channel_space} space")
    if args.morph:
        print("Morphing to common space")
    emeg_values, ch_names, n_reps = load_emeg_pack(emeg_filenames,
                                           emeg_dir=emeg_path,
                                           morph_dir=morph_dir
                                                    if args.morph
                                                    else None,
                                           need_names=True,
                                           ave_mode=args.ave_mode,
                                           inverse_operator_dir=inverse_operator_dir
                                                                if args.use_inverse_operator
                                                                else None,
                                           inverse_operator_suffix= args.inverse_operator_suffix,
                                           p_tshift=None,
                                           snr=args.snr,
                                           )
    
    # emeg_values = emeg_values[:64, :, :]
    # ch_names = ch_names[:64]

    if args.asr_option == 'all' and 'asr' in args.function_path:

        for nn_i in range(0, args.num_neurons):
            # func = load_function(Path(args.base_dir, args.function_path),
            func = load_function(args.function_path,
                                func_name=args.function_name[0],
                                nn_neuron=nn_i,
                                )
        # func = load_function(args.function_path,
        #                      func_name=args.function_name,
        #                      bruce_neurons=(5, 10))
        
            func = func.downsampled(args.downsample_rate)

            if nn_i == 0:
                es = do_gridsearch(
                    emeg_values=emeg_values,
                    channel_names=ch_names,
                    channel_space=channel_space,
                    function=func,
                    seconds_per_split=args.seconds_per_split,
                    n_derangements=args.n_derangements,
                    n_splits=args.n_splits,
                    n_reps=n_reps,
                    start_latency=args.start_latency,
                    plot_location=args.save_plot_location,
                    emeg_t_start=args.emeg_t_start,
                    emeg_sample_rate=args.emeg_sample_rate,
                    audio_shift_correction=args.audio_shift_correction,
                    overwrite=args.overwrite,
                )
            else:
                es += do_gridsearch(
                emeg_values=emeg_values,
                channel_names=ch_names,
                channel_space=channel_space,
                function=func,
                seconds_per_split=args.seconds_per_split,
                n_derangements=args.n_derangements,
                n_splits=args.n_splits,
                n_reps=n_reps,
                start_latency=args.start_latency,
                plot_location=args.save_plot_location,
                emeg_t_start=args.emeg_t_start,
                emeg_sample_rate=args.emeg_sample_rate,
                audio_shift_correction=args.audio_shift_correction,
                overwrite=args.overwrite,
                )

        if args.save_expression_set_location is not None:
            save_expression_set(es, to_path_or_file = Path(args.save_expression_set_location, func.name + '_gridsearch.nkg'), overwrite=args.overwrite)
        expression_plot(es, paired_axes=channel_space == "source", save_to=Path(args.save_plot_location, func.name + '_gridsearch.png'), overwrite=args.overwrite)

    else:

        combined_expression_set = None

        for function_name in args.function_name:
            print(f"Running gridsearch on {function_name}")
            function_values = load_function(Path(base_dir, args.function_path),
                                            func_name=function_name,
                                            bruce_neurons=(5, 10))
            function_values = function_values.downsampled(args.downsample_rate)

            es = do_gridsearch(
                emeg_values=emeg_values,
                channel_names=ch_names,
                channel_space=channel_space,
                function=function_values,
                seconds_per_split=args.seconds_per_split,
                n_derangements=args.n_derangements,
                n_splits=args.n_splits,
                n_reps=n_reps,
                start_latency=args.start_latency,
                plot_location=args.save_plot_location,
                emeg_t_start=args.emeg_t_start,
                audio_shift_correction=audio_shift_correction,
                overwrite=args.overwrite,
            )

            if combined_expression_set is None:
                combined_expression_set = es
            else:
                combined_expression_set += es

        assert combined_expression_set is not None

        if args.save_name is not None and len(args.save_name) > 0:
            combined_names = args.save_name
        elif len(args.function_name) > 2:
            combined_names = f"{len(args.function_name)}_functions"
        else:
            combined_names = "_+_".join(args.function_name)

        if args.save_expression_set_location is not None:
            es_save_path = Path(args.save_expression_set_location, combined_names + '_gridsearch.nkg')
            print(f"Saving expression set to {es_save_path!s}")
            save_expression_set(combined_expression_set, to_path_or_file=es_save_path, overwrite=args.overwrite)

        fig_save_path = Path(args.save_plot_location, combined_names + '_gridsearch.png')
        print(f"Saving expression plot to {fig_save_path!s}")
        expression_plot(combined_expression_set, paired_axes=channel_space == "source", save_to=fig_save_path, overwrite=args.overwrite)

    print(f'Time taken for code to run: {time.time() - start:.4f} s')


if __name__ == '__main__':
    main()
