from pathlib import Path
import argparse
import time
import numpy as np
import subprocess

from kymata.datasets.data_root import data_root_path
from kymata.gridsearch.plain import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
from kymata.io.nkg import save_expression_set
from kymata.plot.plot import expression_plot, plot_top_five_channels_of_gridsearch
from kymata.entities.expression import ExpressionSet, SensorExpressionSet, HexelExpressionSet, p_to_logp, log_base

import re

def submit_job(script_cmd):
    """Submit an sbatch script and return the job ID."""
    result = subprocess.run(script_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Failed to submit job: {result.stderr}")
    # Extract job ID from the output
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise Exception("Failed to find job ID in sbatch output")
    return match.group(1)

def check_job_status(job_id):
    """Check the status of a job by its ID."""
    result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
    print('RESULT')
    print(result)
    return 'invalid' not in result.stdout.lower()


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
    parser.add_argument('--parallel-procs', type=int, default=4,
                        help='split the gridsearch computation across multiple nodes (only used for source space)')
    args = parser.parse_args()
    args.base_dir = Path(args.base_dir)

    emeg_dir = Path(args.base_dir, args.data_path)
    emeg_paths = [Path(emeg_dir, args.emeg_file)]

    participants = ['pilot_01',
                    'pilot_02',
                    'participant_01',
                    'participant_01b',
                    'participant_02',
                    'participant_03',
                    'participant_04',
                    'participant_05',
                    'participant_07',
                    'participant_08',
                    'participant_09',
                    'participant_10',
                    'participant_11',
                    'participant_12',
                    'participant_13',
                    'participant_14',
                    'participant_15',
                    'participant_16',
                    'participant_17'
                    ]

    reps = [f'_rep{i}' for i in range(8)] + ['-ave']

    # emeg_paths = [Path(emeg_dir, p + r) for p in participants[:2] for r in reps[-1:]]

    start = time.time()

    if args.inverse_operator_dir is None:
        inverse_operator = None
    else:
        inverse_operator = Path(args.base_dir, args.inverse_operator_dir, args.inverse_operator_name)

    channel_space = "source" if inverse_operator is not None else "sensor"
    n_reps = len(EMEG_paths) if args.ave_mode == 'add' else 1
    n_samples_per_split = int(args.seconds_per_split * args.emeg_sample_rate * 2 // args.downsample_rate)

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

    if args.parallel_procs > 1: # and channel_space == "source":

        temp_dict_path = 'temp_data_DO_NOT_DELETE.npz'
        result_dict_path = 'temp_result_data_DO_NOT_DELETE' # _${proc_num}.npz
        subproc_file = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/subproc_grid.sh'

        subproc_nchans = ((emeg_values.shape[0] - 1) // args.parallel_procs) + 1
        print('subproc_nchans', subproc_nchans)
        save_dict = {f'emeg_data_{i}': emeg_values[subproc_nchans * i:subproc_nchans * (i + 1)] for i in range(args.parallel_procs)}
        save_dict['args'] = args
        np.savez(temp_dict_path, save_dict)

        job_id = submit_job(["/usr/bin/sbatch", subproc_file, str(args.parallel_procs - 1), temp_dict_path, result_dict_path])

        emeg_values = emeg_values[:subproc_nchans]

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
    )

    if args.parallel_procs > 1 and channel_space == "source":
        
        # Check job status every 30 seconds
        while True:
            if not check_job_status(job_id):
                print(f"Job {job_id} has finished")
                break
            print(f"Job {job_id} is still running...")
            time.sleep(1)

        print(corrs.shape, 'corrs shape')
        print(log_pvalues, 'log_pvals.shape')
        for i in range(1, args.parallel_procs):
            result_dict = np.load(f'{result_dict_path}_{i}.npz')
            print(result_dict)
            log_pvalues = np.concatenate([log_pvalues, result_dict['log_pvalues']], axis=0)
            corrs = np.concatenate([corrs, result_dict['corrs']], axis=0)

        print(corrs.shape, 'corrs shape')
        print(log_pvalues, 'log_pvals.shape')

    latencies_ms = np.linspace(args.start_latency, args.start_latency + (args.seconds_per_split * 1000), n_samples_per_split // 2 + 1)[:-1]
    plot_top_five_channels_of_gridsearch(
                                        corrs=corrs,
                                        auto_corrs=auto_corrs,
                                        function=func,
                                        n_reps=n_reps,
                                        n_splits=args.n_splits,
                                        n_samples_per_split=n_samples_per_split,
                                        latencies=latencies_ms,
                                        save_to=args.save_plot_location,
                                        log_pvalues=log_pvalues,
                                        overwrite=args.overwrite,
                                        )
    if channel_space == "sensor":
        es = SensorExpressionSet(
            functions=func.name,
            latencies=latencies_ms / 1000,  # seconds
            sensors=ch_names,
            data=log_pvalues,
        )
    elif channel_space == "source":
        es = HexelExpressionSet(
            functions=func.name + f"_mirrored-lh",  # TODO: revert to just `function.name` when we have both hemispheres in place
            latencies=latencies_ms / 1000,  # seconds
            hexels=ch_names,
            data_lh=log_pvalues,
            data_rh=log_pvalues,  # TODO: distribute data correctly when we have both hemispheres in place
        )
    else:
        raise NotImplementedError(channel_space)

    if args.save_expression_set_location is not None:
        save_expression_set(es, to_path_or_file = Path(args.save_expression_set_location, args.function_name + '_gridsearch.nkg'), overwrite=args.overwrite)

    expression_plot(es, paired_axes=channel_space == "source", save_to=Path(args.save_plot_location, args.function_name + '_gridsearch.png'), overwrite=args.overwrite)

    print(f'Time taken for code to run: {time.time() - start:.4f}')


if __name__ == '__main__':
    main()
