from pathlib import Path
import argparse
import time
from warnings import warn

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
from kymata.io.nkg import save_expression_set
from kymata.io.yaml import load_config
from kymata.plot.plot import expression_plot

_default_output_dir = Path(data_root_path(), "output")


def main():

    _default_output_dir.mkdir(exist_ok=True, parents=False)

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--emeg-sample-rate', type=int, default=1000,
                        help='sampling rate of the emeg machine (not implemented yet)')
    parser.add_argument('--snr', type=float, default=3, help='inverse solution snr')
    parser.add_argument('--downsample-rate', type=int, default=5, help='downsample_rate')
    parser.add_argument('--base-dir', type=str, default='/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/',
                        help='base data directory')
    parser.add_argument('--emeg-dir', type=str, default='intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data/',
                        help='emeg directory, relative to base dir')
    parser.add_argument('--function-path', type=str, default='predicted_function_contours/GMSloudness/stimulisig', help='location of function stimulisig')
    parser.add_argument('--save-expression-set-location', type=Path, default=Path(_default_output_dir),
                        help="Save the results of the gridsearch into an ExpressionSet .nkg file")
    parser.add_argument('--save-plot-location', type=Path, default=Path(_default_output_dir),
                        help="Save an expression plots, and other plots, in this location")
    parser.add_argument('--overwrite', action="store_true", help="Silently overwrite existing files.")
    parser.add_argument('--function-name', type=str, default="IL", help='function name in stimulisig')
    parser.add_argument('--emeg-file', type=str, default=None, required=False,
                        help='Supply to run only on one participant')
    parser.add_argument('--morph', action="store_true",
                        help="Morph hexel data to fs-average space prior to running gridsearch. "
                             "Only has an effect if an inverse operator is specified.")
    parser.add_argument('--ave-mode', type=str, default="ave",
                        help='either ave or add, either average over the list of repetitions or treat them as extra data')
    parser.add_argument('--inverse-operator-dir', type=str, default=None, help='inverse solution path')
    parser.add_argument('--inverse-operator-name', type=str, default="participant_01_ico5-3L-loose02-cps-nodepth-fusion-inv.fif",
                        help='inverse solution name')
    parser.add_argument('--seconds-per-split', type=float, default=0.5,
                        help='seconds in each split of the recording, also maximum range of latencies being checked')
    parser.add_argument('--n-splits', type=int, default=800,
                        help='number of splits to split the recording into, (set to 400/seconds_per_split for full file)')
    parser.add_argument('--n-derangements', type=int, default=1,
                        help='number of deragements for the null distribution')
    parser.add_argument('--start-latency', type=float, default=-100,
                        help='earliest latency to check in cross correlation')
    parser.add_argument('--emeg-t-start', type=float, default=-200,
                        help='start of the emeg evoked files relative to the start of the function')
    parser.add_argument('--audio-shift-correction', type=float, default=0.000_537_5,
                        help='audio shift correction, for every second of function, add this number of seconds (to the start of the emeg split) per seconds of emeg seen')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

    config = load_config(str(Path(Path(__file__).parent.parent, "kymata", "config", "dataset4.yaml")))

    participants = [
        'pilot_01',
        'pilot_02',
        'participant_01',
        'participant_01b',
        'participant_07',
    ]

    # TODO: move ave-vs-reps choice up to the function interface
    reps = [f'_rep{i}' for i in range(8)] + ['-ave']
    if args.emeg_file is not None:
        emeg_filenames = [args.emeg_file + "-ave"]
    else:
        emeg_filenames = [
            p + r
            for p in participants
            for r in reps[-1:]
        ]

    start = time.time()

    if args.inverse_operator_dir is None:
        inverse_operator = None
    else:
        inverse_operator = Path(args.base_dir, args.inverse_operator_dir, args.inverse_operator_name)

    if (len(emeg_filenames) > 1) and (not args.morph) and (args.ave_mode == "ave") and (inverse_operator is not None):
        warn(f"Averaging without morphing to a common space. "
             f"If you are averaging over multiple participants you should morph to a common space.")

    # Load data
    emeg_path = Path(args.base_dir, args.emeg_dir)
    morph_dir = Path(args.base_dir, "intrim_preprocessing_files", "4_hexel_current_reconstruction", "morph_maps")
    emeg_values, ch_names = load_emeg_pack(emeg_filenames,
                                           emeg_dir=emeg_path,
                                           morph_dir=morph_dir,
                                           use_morph=args.morph,
                                           need_names=True,
                                           ave_mode=args.ave_mode,
                                           inverse_operator=inverse_operator,
                                           p_tshift=None,
                                           snr=args.snr,
                                           )

    func = load_function(Path(args.base_dir, args.function_path),
                         func_name=args.function_name,
                         bruce_neurons=(5, 10))
    func = func.downsampled(args.downsample_rate)

    channel_space = "source" if inverse_operator is not None else "sensor"

    es = do_gridsearch(
        emeg_values=emeg_values,
        channel_names=ch_names,
        channel_space=channel_space,
        function=func,
        seconds_per_split=args.seconds_per_split,
        n_derangements=args.n_derangements,
        n_splits=args.n_splits,
        start_latency=args.start_latency,
        plot_location=args.save_plot_location,
        emeg_t_start=args.emeg_t_start,
        emeg_sample_rate=args.emeg_sample_rate,
        audio_shift_correction=args.audio_shift_correction,
        ave_mode=args.ave_mode,
        overwrite=args.overwrite,
    )

    if args.save_expression_set_location is not None:
        save_expression_set(es, to_path_or_file = Path(args.save_expression_set_location, args.function_name + '_gridsearch.nkg'), overwrite=args.overwrite)

    expression_plot(es, paired_axes=channel_space == "source", save_to=Path(args.save_plot_location, args.function_name + '_gridsearch.png'), overwrite=args.overwrite)

    print(f'Time taken for code to run: {time.time() - start:.4f}')


if __name__ == '__main__':
    main()
