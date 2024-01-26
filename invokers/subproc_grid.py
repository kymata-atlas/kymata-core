from pathlib import Path
import argparse
import time
from numpy import linspace

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
from kymata.io.nkg import save_expression_set
from kymata.plot.plot import expression_plot, plot_top_five_channels_of_gridsearch
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet, p_to_logp, log_base

_default_output_dir = Path(data_root_path(), "output")


def main():

    _default_output_dir.mkdir(exist_ok=True, parents=False)

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--emeg-sample-rate', type=int, default=1000,
                        help='sampling rate of the emeg machine (not implemented yet)')
    parser.add_argument('--snr', type=float, default=3, help='inverse solution snr')
    parser.add_argument('--downsample-rate', type=int, default=5, help='downsample_rate')
    parser.add_argument('--base-dir', type=str, required=True, help='base data directory')
    parser.add_argument('--data-path', type=str, required=True, help='data path after base dir')
    parser.add_argument('--function-path', type=str, required=True, help='location of function stimulisig')
    parser.add_argument('--save-expression-set-location', type=Path, default=Path(_default_output_dir),
                        help="Save the results of the gridsearch into an ExpressionSet .nkg file")
    parser.add_argument('--save-plot-location', type=Path, default=Path(_default_output_dir),
                        help="Save an expression plots, and other plots, in this location")
    parser.add_argument('--overwrite', action="store_true", help="Silently overwrite existing files.")
    parser.add_argument('--function-name', type=str, required=True, help='function name in stimulisig')
    parser.add_argument('--emeg-file', type=str, required=True, help='emeg_file_name')
    parser.add_argument('--ave-mode', type=str, default="ave",
                        help='either ave or add, either average over the list of repetitions or treat them as extra data')
    parser.add_argument('--inverse-operator-dir', type=str, required=False, default=None, help='inverse solution path')
    parser.add_argument('--inverse-operator-name', type=str, default="participant_01_ico5-3L-loose02-cps-nodepth.fif",
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
    parser.add_argument('--parallel-splits', type=int, default=4,
                        help='split the gridsearch computation across multiple nodes (only used for source space)')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

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
    n_reps = len(EMEG_paths) if args.ave_mode == 'add' else 1
    n_samples_per_split = int(args.seconds_per_split * args.emeg_sample_rate * 2 // args.downsample_rate)

    log_pvalues, corrs, auto_corrs = do_gridsearch(
        emeg_values=emeg_values,
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


    print(f'Time taken for code to run: {time.time() - start:.4f}')


if __name__ == '__main__':
    main()
