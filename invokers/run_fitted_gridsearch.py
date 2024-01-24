from pathlib import Path

import sys
sys.path.append('/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox')

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

import argparse
from numpy.typing import NDArray
from kymata.gridsearch.gridsearch import do_gridsearch
from kymata.io.functions import load_function
from kymata.io.mne import load_emeg_pack
from kymata.math.vector import normalize
# from kymata.plot.plotting import expression_plot


def main():

    parser = argparse.ArgumentParser(description='Gridsearch Params')
    parser.add_argument('--emeg_sample_rate', type=int, default=1000,
                        help='sampling rate of the emeg machine (always 1000 I thought?)')
    parser.add_argument('--snr', type=float, default=3,
                        help='inverse solution snr')
    parser.add_argument('--downsample_rate', type=int, default=5,
                        help='downsample_rate')
    parser.add_argument('--base_dir', type=str, default="/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/",
                        help='base data directory')
    parser.add_argument('--data_path', type=str, default="intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data",
                        help='data path after base dir')
    parser.add_argument('--function_path', type=str, default="predicted_function_contours/GMSloudness/stimulisig",
                        help='snr')
    parser.add_argument('--function_name', type=str, default="d_IL2",
                        help='function name in stimulisig')
    parser.add_argument('--emeg_file', type=str, default="participant_01-ave",
                        help='emeg_file_name')
    parser.add_argument('--ave_mode', type=str, default="ave",
                        help='either ave or add, either average over the list of repetitions or treat them as extra data')
    parser.add_argument('--inverse_operator', type=str, default="intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators",
                        help='inverse solution path')
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

    emeg_paths = [Path(emeg_dir, p + r) for p in participants[:2] for r in reps[-1:]]

    # inverse_operator = Path(args.base_dir, args.inverse_operator, f"{participants[0]}_ico5-3L-loose02-cps-nodepth.fif")

    ### LOAD FUNCTION ###

    args.n_splits = 801

    # args.function_path = 'predicted_function_contours/Bruce_model/neurogramResults'
    # args.function_name = 'neurogram_mr'
    args.function_path = 'predicted_function_contours/asr_models/w2v_convs'
    args.function_name = 'conv_layer3'
    func_name = args.function_name


    func_dict = np.load(Path(args.base_dir, args.function_path).with_suffix(".npz"))
    func = func_dict[func_name][:, :400_000]

    ### LOAD EMEG ###
    emeg, ch_names = load_emeg_pack(emeg_paths,
                                    need_names=False,
                                    ave_mode=args.ave_mode,
                                    inverse_operator=None,
                                    p_tshift=None,
                                    snr=args.snr)

    # print(emeg.shape) # (370, 1, 402001) 
    n_channels = emeg.shape[0]
    n_reps = 1
    n_samples_per_split = int(1000 * args.seconds_per_split)

    # Reshape EMEG into splits of `seconds_per_split` s
    split_initial_timesteps = [int(args.start_latency + round(i * 1000 * args.seconds_per_split * (1 + args.audio_shift_correction)) - args.emeg_t_start)
        for i in range(args.n_splits)]

    emeg_reshaped = np.zeros((n_channels, args.n_splits * n_reps, n_samples_per_split))
    for j in range(n_reps):
        for split_i in range(args.n_splits):
            split_start = split_initial_timesteps[split_i]
            split_stop = split_start + int(1000 * args.seconds_per_split)
            emeg_reshaped[:, split_i + (j * args.n_splits), :] = emeg[:, j, split_start:split_stop]

    del emeg

    #print(emeg_reshaped.shape)  # (370, 800, 500)
    #print(func.shape)           # (40, 400_000)

    def corr(x, y):
        return np.sum(normalize(x) * normalize(y))

    latency = 145
    channel = 209

    func = normalize(func) * 700

    _emeg_reshaped = emeg_reshaped.reshape(370, 400_500)
    ridge_model = Ridge(alpha=1e4, positive=False)

    for latency in range(0, 500, 5):

        emeg_reshaped = _emeg_reshaped[:, latency:400_000 + latency]

        #plt.plot(normalize(emeg_reshaped[209,:2000]))
        #plt.plot(normalize(func[10,:2000]))
        #plt.plot(normalize(func[11,:2000]))
        #plt.savefig('example_2.png')
        #return

        emeg_209 = emeg_reshaped[channel]

        emeg_209 = normalize(emeg_209) * 700

        split = 0.8
        split = int(split * 400_000)

        # Fit Ridge Regression
        ridge_model.fit(func[:, :split].T, emeg_209[:split])

        print('latency:', latency)
        print('train r^2:', ridge_model.score(func[:, :split].T, emeg_209[:split])) # = corr^2
        print('val r^2:  ', ridge_model.score(func[:, split:].T, emeg_209[split:])) # = corr^2
        print()


    """func_ = np.mean(func[5:10], axis=0)

    n_splits = 800
    r_func = func_.reshape(n_splits, -1)
    r_emeg = emeg_reshaped.reshape(n_channels, n_splits, -1)

    for i in (207, 209, 210):
        print(i,     corr(func_[:split], emeg_reshaped[i, :split]))
        print('   ', corr(func_[split:], emeg_reshaped[i, split:]))
        print(' - ', np.mean([corr(r_func[j], r_emeg[i, j]) for j in range(n_splits)]))"""

    # preds = ridge_model.predict(func.T)
    # print(np.sum(normalize(preds) * normalize(emeg_209)))

    #print(ridge_model.coef_)
    #print(ridge_model.intercept_)
    #print(ridge_model.get_params())

    plt.plot(ridge_model.coef_)
    plt.savefig('example_2.png')


if __name__ == '__main__':
    main()
