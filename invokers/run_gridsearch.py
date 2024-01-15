from pathlib import Path
import argparse

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
from kymata.io.nkg import save_expression_set
from kymata.plot.plot import expression_plot

_default_output_dir = Path(data_root_path(), "output")


def main():

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--emeg-sample-rate', type=int, default=1000,
                        help='sampling rate of the emeg machine (not implemented yet)')
    parser.add_argument('--snr', type=float, default=3, help='inverse solution snr')
    parser.add_argument('--downsample-rate', type=int, default=5, help='downsample_rate')
    parser.add_argument('--base-dir', type=str, required=True, help='base data directory')
    parser.add_argument('--data-path', type=str, required=True, help='data path after base dir')
    parser.add_argument('--function-path', type=str, required=True, help='location of function stimulisig')
    parser.add_argument('--save-expression-set', type=Path, default=Path(_default_output_dir, "gridsearch.nkg"),
                        help="Save the results of the gridsearch into an ExpressionSet .nkg file")
    parser.add_argument('--save-plot', type=Path, default=Path(_default_output_dir, "gridsearch.png"),
                        help="Save an expression plot file")
    parser.add_argument("--reality-check-plot-location", type=Path, default=Path(_default_output_dir))
    parser.add_argument("--reality-check-plot-name", type=str, required=False, default=None,
                        help="Save reality-check plots")
    parser.add_argument("--add-autocorr", action="store_true", help="Adds autocorrelation to reality-check plots")
    parser.add_argument('--overwrite', action="store_true", help="Silently overwrite existing files.")
    parser.add_argument('--function-name', type=str, required=True, help='function name in stimulisig')
    parser.add_argument('--emeg-file', type=str, required=True, help='emeg_file_name')
    parser.add_argument('--ave-mode', type=str, default="ave",
                        help='either ave or add, either average over the list of repetitions or treat them as extra data')
    parser.add_argument('--inverse-operator', type=str, required=False, default=None, help='inverse solution path')
    parser.add_argument('--seconds-per-split', type=float, default=0.5,
                        help='seconds in each split of the recording, also maximum range of latencies being checked')
    parser.add_argument('--n-splits', type=int, default=800,
                        help='number of splits to split the recording into, (set to 400/seconds_per_split for full file)')
    parser.add_argument('--n-derangements', type=int, default=1,
                        help='inverse solution snr')
    parser.add_argument('--start-latency', type=float, default=-100,
                        help='earliest latency to check in cross correlation')
    parser.add_argument('--emeg-t-start', type=float, default=-200,
                        help='start of the emeg evoked files relative to the start of the function')
    parser.add_argument('--audio-shift-correction', type=float, default=0.000_537_5,
                        help='audio shift correction, for every second of function, add this number of seconds (to the start of the emeg split) per seconds of emeg seen')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

    emeg_dir = Path(args.base_dir, args.data_path)
    emeg_paths = [Path(emeg_dir, args.emeg_file)]

    participants = ['participant_01',
                    'participant_01b',
                    'participant_02',
                    'participant_03',
                    'participant_04',
                    'participant_05',
                    'pilot_01',
                    'pilot_02']

    reps = [f'_rep{i}' for i in range(8)] + ['-ave']

    # emeg_paths = [Path(emeg_dir, p + r) for p in participants[:2] for r in reps[-1:]]

    if args.inverse_operator is None:
        inverse_operator = None
    else:
        inverse_operator = Path(args.base_dir, args.inverse_operator, f"{participants[0]}_ico5-3L-loose02-cps-nodepth.fif")

    # Load data
    emeg_values, ch_names = load_emeg_pack(emeg_paths,
                                           need_names=True,
                                           ave_mode=args.ave_mode,
                                           inverse_operator=inverse_operator,
                                           p_tshift=None,
                                           snr=args.snr)

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
        emeg_t_start=args.emeg_t_start,
        emeg_sample_rate=args.emeg_sample_rate,
        audio_shift_correction=args.audio_shift_correction,
        ave_mode=args.ave_mode,
        add_autocorr=args.add_autocorr,
        plot_name=args.reality_check_plot_name,
        plot_location=args.reality_check_plot_location,
    )

    if args.save_expression_set is not None:
        save_expression_set(es, args.save_expression_set, overwrite=args.overwrite)

    expression_plot(es, paired_axes=channel_space == "source", save_to=args.save_plot, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
