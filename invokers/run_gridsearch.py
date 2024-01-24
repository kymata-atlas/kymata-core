from pathlib import Path

import sys
sys.path.append('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox')

import numpy as np
import argparse
from numpy.typing import NDArray
from kymata.gridsearch.gridsearch import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
# from kymata.plot.plotting import expression_plot
import time
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--emeg_sample_rate', type=int, default=1000,
                        help='sampling rate of the emeg machine (not implemented yet)')
    parser.add_argument('--snr', type=float, default=3,
                        help='inverse solution snr')
    parser.add_argument('--downsample_rate', type=int, default=5,
                        help='downsample_rate')
    parser.add_argument('--base_dir', type=str, default="/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/",
                        help='base data directory')
    parser.add_argument('--data_path', type=str, default="intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data",
                        help='data path after base dir')
    parser.add_argument('--function_path', type=str, default="predicted_function_contours/GMSloudness/stimulisig",
                        help='function path, e.g. predicted_function_contours/GMSloudness/stimulisig')
    parser.add_argument('--function_name', type=str, default="d_IL2",
                        help='function name in stimulisig')
    parser.add_argument('--emeg_file', type=str, default="participant_01-ave",
                        help='emeg_file_name')
    parser.add_argument('--ave_mode', type=str, default="ave",
                        help='either ave or add, either average over the list of repetitions or treat them as extra data')
    parser.add_argument('--inverse_operator_path', type=str, default="intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators",
                        help='inverse solution path')
    parser.add_argument('--inverse_operator_name', type=str, default="participant_01_ico5-3L-loose02-cps-nodepth.fif",
                        help='inverse solution name')
    parser.add_argument('--seconds_per_split', type=float, default=0.5,
                        help='seconds in each split of the recording, also maximum range of latencies being checked')
    parser.add_argument('--n_splits', type=int, default=800,
                        help='number of splits to split the recording into, (set to 400/seconds_per_split for full file)')
    parser.add_argument('--n_derangements', type=int, default=1,
                        help='inverse solution snr')
    parser.add_argument('--start_latency', type=float, default=-100,
                        help='earliest latency to check in cross correlation')
    parser.add_argument('--emeg_t_start', type=float, default=-200,
                        help='start of the emeg evoked files relative to the start of the function')
    parser.add_argument('--audio_shift_correction', type=float, default=0.000_537_5,
                        help='audio shift correction, for every second of function, add this number of seconds (to the start of the emeg split) per seconds of emeg seen')
    parser.add_argument('--eeg_meg_only', type=str, default='both',
                        help="must be one of ('eeg_only', 'meg_only', 'both')")                        
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
                    "participant_07",
                    "participant_08",
                    "participant_09",
                    "participant_10",
                    "participant_11",
                    "participant_12",
                    "participant_13",
                    "participant_14",
                    "participant_15",
                    "participant_16",
                    "participant_17",
                    'pilot_01',
                    'pilot_02']

    reps = [f'_rep{i}' for i in range(8)] + ['-ave']

    # emeg_paths = [Path(emeg_dir, p + r) for p in participants[:2] for r in reps[-1:]]

    if args.inverse_operator_name.lower() == 'none':
        inverse_operator = None
    else:
        inverse_operator = Path(args.base_dir, args.inverse_operator_path, args.inverse_operator_name)
    # inverse_operator = Path('/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators/meg15_0045_ico-5-3L-loose02-diagnoise-nodepth-reg-inv-csd.fif')


    # args.function_path = 'predicted_function_contours/Bruce_model/neurogramResults'
    # args.function_name = 'neurogram_mr'
    # args.function_path = 'predicted_function_contours/asr_models/w2v_convs'
    # args.function_name = 'conv_layer6'

    # args.n_splits = 400
    # args.seconds_per_split = 1

    # Load data
    import os
    if not os.path.exists(str(emeg_paths[0]) + '.fif'):
        # print(emeg_paths)
        import sys
        sys.exit(1)
    
    t0 = time.time()

    emeg, ch_names = load_emeg_pack(emeg_paths,
                                    need_names=False,
                                    ave_mode=args.ave_mode,
                                    inverse_operator=inverse_operator, #inverse_operator, # set to None/inverse_operator if you want to run on sensor space/source space
                                    p_tshift=None,
                                    snr=args.snr,
                                    eeg_meg_only=args.eeg_meg_only,
                                    )

    # args.function_name = 'd_IL2'

    # d_STL = load_function(Path(args.base_dir, args.function_path), func_name='d_STL').downsampled(args.downsample_rate)
    func = load_function(Path(args.base_dir, args.function_path), func_name=args.function_name).downsampled(args.downsample_rate)

    import sys; sys.stdout.flush()

    es = do_gridsearch(
        emeg_values=emeg,
        sensor_names=ch_names,
        function=func,
        seconds_per_split=args.seconds_per_split,
        n_derangements=args.n_derangements,
        n_splits=args.n_splits,
        start_latency=args.start_latency,
        emeg_t_start=args.emeg_t_start,
        emeg_sample_rate=args.emeg_sample_rate,
        audio_shift_correction=args.audio_shift_correction,
        ave_mode=args.ave_mode,
        part_name=args.emeg_file,
        )

    print(f'Time elapsed: {time.time() - t0:.4f}')

    """function_names = ['neurogram_mr',
                      'd_IL1',
                      'd_IL2',
                      'd_IL3',
                      'd_STL2',
                      'd_STL3',
                      'd_STL',
                      'IL',
                      'STL',
                      ][:]"""
    """function_names = ['neurogram_mr',
                      'IL1',
                      'IL2',
                      'IL3',
                      'STL1',
                      'STL2',
                      'STL3',
                      'IL',
                      'STL',
                      ]
    function_paths = ['predicted_function_contours/Bruce_model/neurogramResults'] + \
                     ['predicted_function_contours/GMSloudness/stimulisig'] * (len(function_names) - 1)
    #function_paths = ['predicted_function_contours/GMSloudness/stimulisig'] * len(function_names)

    n_funcs = len(function_names)

    sensor_best_funcs = []
    best_pvalues = []
    best_lats = []


    for i in range(n_funcs):

        args.function_path = function_paths[i]
        args.function_name = function_names[i]

        func = load_function(Path(args.base_dir, args.function_path),
                            func_name=args.function_name,
                            bruce_neurons=(5, 10))
        func = func.downsampled(args.downsample_rate)

        es = do_gridsearch(
            emeg_values=emeg,
            sensor_names=ch_names,
            function=func,
            seconds_per_split=args.seconds_per_split,
            n_derangements=args.n_derangements,
            n_splits=args.n_splits,
            start_latency=args.start_latency,
            emeg_t_start=args.emeg_t_start,
            emeg_sample_rate=args.emeg_sample_rate,
            audio_shift_correction=args.audio_shift_correction,
            ave_mode=args.ave_mode,
            plot_name=None,
        )

        # expression_plot(es)

        sensor_pvalues, sensor_lats = es

        if len(sensor_best_funcs) == 0:
            sensor_best_funcs = np.full(sensor_pvalues.shape, i)
            best_pvalues = sensor_pvalues
            best_lats = sensor_lats
        else:
            sensor_best_funcs[sensor_pvalues > best_pvalues] = i
            best_lats[sensor_pvalues > best_pvalues] = sensor_lats[sensor_pvalues > best_pvalues]
            best_pvalues[sensor_pvalues > best_pvalues] = sensor_pvalues[sensor_pvalues > best_pvalues]

    # print(sensor_best_funcs)
    # print(best_pvalues)
    # print(best_lats)

    top_lats = []
    top_pvals = []
    for j in range(n_funcs):
        top_lats.append(str(best_lats[sensor_best_funcs==j][np.argmax(best_pvalues[sensor_best_funcs==j])]))
        top_pvals.append(str(best_pvalues[sensor_best_funcs==j][np.argmax(best_pvalues[sensor_best_funcs==j])]))
    print(args.emeg_file+','+','.join(top_lats)+','+','.join(top_pvals))"""


    """top_lats = []
    top_pvals = []
    part_num = []
    for i in open('result_exp.txt', 'r'):
        row = i.split(',')
        if row[0] != 'emeg_file':
            top_lats.append([float(i) for i in row[1:n_funcs+1]])
            top_pvals.append([float(i) for i in row[n_funcs+1:]])
            part_num.append(row[0])

    print(part_num)
    [print(i) for i in top_lats]
    [print(i) for i in top_pvals]"""

    """plt.figure(3)
    plt.xlim(args.start_latency, args.seconds_per_split * 1000)
    plt.axvline(0, color='k')
    for j in range(n_funcs):
        plt.plot(best_lats[sensor_best_funcs==j], best_pvalues[sensor_best_funcs==j], '.', label=function_names[j])

    plt.legend()
    plt.savefig(f'example_3.png')
    # plt.savefig(f'exp_plot_{args.emeg_file}.png')"""


if __name__ == '__main__':
    main()
