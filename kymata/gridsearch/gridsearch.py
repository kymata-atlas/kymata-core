import numpy as np
import matplotlib.pyplot as plt
import mne
import time
import os.path
import h5py
from scipy.stats import ttest_ind
import scipy.stats as stats

def norm(x):
    x -= np.mean(x, axis=-1, keepdims=True)
    x /= np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
    return x

def get_EMEG_data(EMEG_path, need_names=False):
    if not os.path.isfile(f'{EMEG_path}.npy') or need_names:
        evoked = mne.read_evokeds(f'{EMEG_path}.fif', verbose=False)  # should be len 1 list
        EMEG = evoked[0].get_data()  # numpy array shape (sensor_num, N) = (370, 406_001)
        EMEG /= np.max(EMEG, axis=1, keepdims=True)
        np.save(f'{EMEG_path}.npy', np.array(EMEG, dtype=np.float16))  # nb fif is float32
        return EMEG, evoked[0].ch_names
    else:
        return np.load(f'{EMEG_path}.npy'), None

def load_function(function_path, func_name, downsample_rate):
    if not os.path.isfile(function_path + '.h5'):
        import scipy.io
        mat = scipy.io.loadmat(function_path + '.mat')['stimulisig']
        with h5py.File(function_path + '.h5', 'w') as f:
            for key in mat.dtype.names:
                if key != 'name':
                    f.create_dataset(key, data=np.array(mat[key][0,0], dtype=np.float16))
        func = np.array(mat[func_name][0][0])[:,::downsample_rate]  # shape = (400, 1000)
    else:
        with h5py.File(function_path + '.h5', 'r') as f:
            func = f[func_name][:,::downsample_rate]
    return func.flatten()

def generate_derangement(n):  # approx 3ms runtime for n=400
    while True:
        v = np.arange(n)
        for j in range(n - 1, -1, -1):
            p = np.random.randint(0, j + 1)
            if v[p] == j:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            return v

def ttest(corrs, f_alpha=0.001, use_all_lats=True):
    # Vectorised Welch's t-test
    # Fisher Z-Transformation
    corrs = 0.5 * np.log((1 + corrs) / (1 - corrs))
    n_channels, nDerangements, n_splits, T_steps = corrs.shape

    true_mean = np.mean(corrs[:, 0, :, :], axis=1)
    true_var = np.var(corrs[:, 0, :, :], axis=1, ddof=1)
    true_n = n_splits
    if use_all_lats:
        rand_mean = np.mean(corrs[:, 1:, :, :].reshape(n_channels, -1), axis=1).reshape(n_channels, 1)
        rand_var = np.var(corrs[:, 1:, :, :].reshape(n_channels, -1), axis=1, ddof=1).reshape(n_channels, 1)
        rand_n = n_splits * nDerangements * T_steps
    else:
        rand_mean = np.mean(corrs[:, 1:, :, :].reshape(n_channels, -1, T_steps), axis=1)
        rand_var = np.var(corrs[:, 1:, :, :].reshape(n_channels, -1, T_steps), axis=1, ddof=1)
        rand_n = n_splits * nDerangements

    # Vectorized two-sample t-tests for all channels and time steps
    numerator = true_mean - rand_mean
    denominator = np.sqrt(true_var / true_n + rand_var / rand_n)
    df = ((true_var / true_n + rand_var / rand_n) ** 2 /
        ((true_var / true_n) ** 2 / (true_n - 1) +
        (rand_var / rand_n) ** 2 / (rand_n - 1)))

    t_stat = numerator / denominator
    p = stats.t.sf(np.abs(t_stat), df) * 2  # two-tailed p-value

    # Adjust p-values for multiple comparisons (Bonferroni correction) [NOT SURE ABOUT THIS]
    pvalues_adj = p #np.minimum(1, p * T_steps * n_channels / (1 - f_alpha))

    return pvalues_adj

def do_gridsearch(data_root='/imaging/projects/cbu/kymata/data',
                  data_path='/dataset_4-english-narratives/intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data',
                  function_root='/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/predicted_function_contours',
                  function_path='/GMSloudness/stimulisig',
                  func_name='d_IL2',
                  audio_shift_correction=0.5375,
                  seconds_per_split=0.5,
                  n_splits=800,
                  downsample_rate=5,
                  nDerangements=1,
                  rep='-ave',
                  save_pvalues_path=None,
                  p='participant_01',
                  start_lat=-100):
    '''Do the Kymata gridsearch over all hexels for all latencies'''

    EMEG_path = f'{data_root}/{data_path}/{p}{rep}'
    t0 = time.time()

    # Load data
    T_steps = int((2000 * seconds_per_split) // downsample_rate)
    EMEG, EMEG_names = get_EMEG_data(EMEG_path, save_pvalues_path)
    func = load_function(function_root + function_path, 
                         func_name=func_name,
                         downsample_rate=downsample_rate)
    func = func.reshape(n_splits, T_steps // 2)
    n_channels = EMEG.shape[0]

    t1 = time.time(); print(f'Load time: {round(t1-t0, 2)}s')

    # Reshape EMEG into splits of 'seconds_per_split' s
    second_start_points = [start_lat + 200 + round((1000 + audio_shift_correction) * seconds_per_split * i) for i in range(n_splits)]  # correcting for audio shift in delivery
    R_EMEG = np.zeros((n_channels, n_splits, T_steps))
    for i in range(n_splits):
        R_EMEG[:,i,:] = EMEG[:, second_start_points[i]:second_start_points[i] + int(2000 * seconds_per_split):downsample_rate]

    # Get derangement for null dist:
    permorder = np.zeros((nDerangements, n_splits), dtype=int)
    for i in range(nDerangements):
        permorder[i, :] = generate_derangement(n_splits)
    permorder = np.vstack((np.arange(n_splits), permorder))

    # FFT cross-corr
    R_EMEG = np.fft.rfft(norm(R_EMEG), n=T_steps, axis=-1)
    F_func = np.conj(np.fft.rfft(norm(func), n=T_steps, axis=-1))
    corrs = np.zeros((n_channels, nDerangements+1, n_splits, T_steps//2))
    for i, order in enumerate(permorder):
        deranged_EMEG = R_EMEG[:, order, :]
        corrs[:, i] = np.fft.irfft(deranged_EMEG * F_func)[:, :, :T_steps//2]

    t2 = time.time(); print(f'Corr time: {round(t2-t1, 2)}s')

    plt.figure(1)
    plt.plot(np.linspace(start_lat, start_lat + 1000 * seconds_per_split, T_steps//2), np.mean(corrs[[207, 210, 5, 10], 0], axis=-2).T)
    plt.plot(np.linspace(start_lat, start_lat + 1000 * seconds_per_split, T_steps//2), np.mean(corrs[0:300:15, 1], axis=-2).T, 'cyan')
    plt.axvline(0, color='k')
    plt.savefig('testing_savefig.png')
    plt.clf()

    t2 = time.time()

    pvalues = ttest(corrs)

    t3 = time.time(); print(f'T-Test time: {round(t3-t2,2)}s' )
    
    plt.figure(2)
    plt.plot(np.linspace(start_lat, start_lat + 1000 * seconds_per_split, T_steps//2), -np.log(pvalues[[207,210,5,10]].T))
    plt.axvline(0, color='k')
    plt.savefig('testing_fig2.png')
    plt.clf()

    if save_pvalues_path:
        with h5py.File(save_pvalues_path + '.h5', 'w') as f:
            f.create_dataset('pvalues', data=pvalues)
            f.create_dataset('latencies', data=np.linspace(start_lat, int(start_lat + (1000 * seconds_per_split)), 1000//downsample_rate))
            f.create_dataset('EMEG_names', data=EMEG_names)

    return


if __name__ == "__main__":
    do_gridsearch()
