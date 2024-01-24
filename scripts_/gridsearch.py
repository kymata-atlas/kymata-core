import numpy as np
import matplotlib.pyplot as plt
import time
import os.path
import h5py
from scipy.stats import ttest_ind
import scipy.stats as stats
import mne
import sys
import argparse

def norm(x):
    x -= np.mean(x, axis=-1, keepdims=True)
    x /= np.sqrt(np.sum(x**2, axis=-1, keepdims=True))
    return x

def get_EMEG_data(EMEG_path, need_names=False):
    if not os.path.isfile(f'{EMEG_path}.npy') or need_names:
        evoked = mne.read_evokeds(f'{EMEG_path}.fif', verbose=False)  # should be len 1 list
        EMEG = evoked[0].get_data()  # numpy array shape (sensor_num, N) = (370, 406_001)
        EMEG /= np.max(EMEG, axis=1, keepdims=True)
        if not os.path.isfile(f'{EMEG_path}.npy'):
            np.save(f'{EMEG_path}.npy', np.array(EMEG, dtype=np.float16))  # nb fif is float32
        return EMEG, evoked[0].ch_names
    else:
        return np.load(f'{EMEG_path}.npy'), None
    
def get_ave_EMEG_data(EMEG_paths, need_names=False, ave_mode=None, inverse_operator=None, p_tshift=None):  # TODO: FIX PRE-AVE-NORMALISATION
    if p_tshift is None:
        p_tshift = [0]*len(EMEG_paths)
    EMEG, EMEG_names = get_EMEG_data(EMEG_paths[0], need_names)
    EMEG = EMEG[:,p_tshift[0]:402001 + p_tshift[0]]
    EMEG = np.expand_dims(EMEG, 1)
    if ave_mode == 'add':
        for i in range(1, len(EMEG_paths)):
            t_shift = p_tshift[i]
            new_EMEG = get_EMEG_data(EMEG_paths[i])[0][:,t_shift:402001 + t_shift]
            EMEG = np.concatenate((EMEG, np.expand_dims(new_EMEG, 1)), axis=1)
    elif ave_mode == 'ave':
        for i in range(1, len(EMEG_paths)):
            t_shift = p_tshift[i]
            EMEG += np.expand_dims(get_EMEG_data(EMEG_paths[i])[0][:,t_shift:402001 + t_shift], 1)
    else:
        raise NotImplementedError(f'ave_mode "{ave_mode}" not known')

    return EMEG, EMEG_names

def get_source_data(EMEG_paths, inverse_operator, need_names=False, ave_mode=None, snr=4): # TODO:  FIX FOR AVERAGING
    lambda2 = 1.0 / snr ** 2
    for EMEG_path in EMEG_paths:
        inverse_operator = mne.minimum_norm.read_inverse_operator(inverse_operator) #, verbose=False)
        evoked = mne.read_evokeds(f'{EMEG_path}.fif', verbose=False)[0]
        mne.set_eeg_reference(evoked, projection=True) #, verbose=False)
        stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, 'MNE', pick_ori='normal') #, verbose=False)
        return np.expand_dims(stc.lh_data, 1), evoked.ch_names

def load_function(function_path, func_name, downsample_rate, n_derivatives=0, bruce_neurons=(5,10)):
    if not os.path.isfile(function_path + '.h5'):
        import scipy.io
        if 'neurogram' in func_name:
            print('USING BRUCE MODEL')
            mat = scipy.io.loadmat(function_path + '.mat')['neurogramResults']
            func = np.array(mat[func_name][0][0])
            func = np.mean(func[bruce_neurons[0]:bruce_neurons[1]], axis=0)
            tt = np.array(mat['t_'+func_name[-2:]][0][0])

            T_end = tt[0, -1]
            if func_name == 'neurogram_mr':
                func = np.interp(np.linspace(0, 400, 400_000 + 1)[:-1], tt[0], func)
            elif func_name == 'neurogram_ft':
                func = np.cumsum(func * (tt[0, 1] - tt[0, 0]), axis=-1)
                func = np.interp(np.linspace(0, 400, 400_000 + 1)[:], tt[0], func)
                func = np.diff(func, axis=-1)

        else:
            mat = scipy.io.loadmat(function_path + '.mat')['stimulisig']
            with h5py.File(function_path + '.h5', 'w') as f:
                for key in mat.dtype.names:
                    if key != 'name':
                        f.create_dataset(key, data=np.array(mat[key][0,0], dtype=np.float16))
            
            func = np.array(mat[func_name][0][0]) # shape = (400, 1000)
    else:
        with h5py.File(function_path + '.h5', 'r') as f:
            func = np.array(f[func_name])
    
    if func_name in ('STL', 'IL', 'LTL'):
        func = func.T
    func = func.flatten()

    for _ in range(n_derivatives):
        func = np.convolve(func, [-1, 1], 'same')  # derivative

    return func[::downsample_rate]

def generate_derangement(n, mod=int(1e9)):  # approx 3ms runtime for n=400
    while True:
        v = np.arange(n)
        for j in range(n - 1, -1, -1):
            p = np.random.randint(0, j + 1)
            if v[p] % mod == j % mod:
                break
            else:
                v[j], v[p] = v[p], v[j]
        else:
            return v

def ttest(corrs, f_alpha=0.001, use_all_lats=True):
    # Vectorised Welch's t-test
    # nb:  I think use_all_lats should probably be switched off in the end, 
    #      some autocorrelation in the null dist latency-wise so not all draws
    #      are independent where the test distribution don't have this effect
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

    if np.min(df) <= 300:
        log_p = np.log(stats.t.sf(np.abs(t_stat), df) * 2)  # two-tailed p-value
    else:
        # norm v good approx for this, (logsf for t not implemented in logspace)
        log_p = stats.norm.logsf(np.abs(t_stat)) + np.log(2) 

    return log_p

def do_gridsearch(data_root='/imaging/projects/cbu/kymata/data',
                  data_path='/dataset_4-english-narratives/intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data',
                  inverse_operator='/dataset_4-english-narratives/intrim_preprocessing_files/4_hexel_current_reconstruction/inverse-operators',
                  function_root='/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/predicted_function_contours',
                  function_path='/GMSloudness/stimulisig',
                  func_name='d_IL2',
                  audio_shift_correction=0.5375,
                  seconds_per_split=0.5,
                  n_splits=800,
                  downsample_rate=1,
                  nDerangements=1,
                  rep='-ave',
                  save_pvalues_path=None,
                  p='participant_01',
                  p_tshift=None,
                  start_lat=-100,
                  plot_name='example',
                  verbose_timing=False,
                  ave_mode='add',  # 'add' or 'ave' for dealing with extra reps
                  snr=4,
                  add_autocorr=True,
                  ):
    '''Do the Kymata gridsearch over all hexels for all latencies'''

    if verbose_timing: t0 = time.time()

    if type(rep) != list:
        rep = [rep]
    if type(p) != list:
        p = [p]
    EMEG_paths = [f'{data_root}{data_path}/{part}{r}' for r in rep for part in p]

    if inverse_operator is not None:
        inverse_operator = f'{data_root}{inverse_operator}/{p}_ico5-3L-loose02-cps-nodepth.fif'

    # Load data
    T_steps = int((2000 * seconds_per_split) // downsample_rate)
    if inverse_operator:
        EMEG, EMEG_names = get_source_data(EMEG_paths, inverse_operator, snr=snr)
    else:
        EMEG, EMEG_names = get_ave_EMEG_data(EMEG_paths, 
                                             need_names=save_pvalues_path,
                                             ave_mode=ave_mode,
                                             p_tshift=p_tshift)
    func = load_function(function_root + function_path, 
                         func_name=func_name,
                         downsample_rate=downsample_rate)

    if ave_mode == 'add':
        n_reps = len(EMEG_paths)
    else:
        n_reps = 1
    func_length = n_splits * n_reps * T_steps // 2
    if func_length < func.shape[0]:
        func = func[:func_length].reshape(n_splits, T_steps // 2)
        print(f'WARNING: not using full 400s of the file (only using {round(n_splits * seconds_per_split, 2)}s)')
    else:
        func = func.reshape(n_splits, T_steps // 2)
    n_channels = EMEG.shape[0]

    if verbose_timing: t1 = time.time(); print(f'Load time: {round(t1-t0, 2)}s'); sys.stdout.flush()

    # Reshape EMEG into splits of 'seconds_per_split' s
    second_start_points = [start_lat + 200 + round((1000 + audio_shift_correction) * seconds_per_split * i) for i in range(n_splits)]  # correcting for audio shift in delivery
    R_EMEG = np.zeros((n_channels, n_splits * n_reps, T_steps))
    for j in range(n_reps):
        for i in range(n_splits):
            R_EMEG[:, i + (j * n_splits), :] = EMEG[:, j, second_start_points[i]:second_start_points[i] + int(2000 * seconds_per_split):downsample_rate]

    del EMEG

    # Get derangement for null dist:
    permorder = np.zeros((nDerangements, n_splits * n_reps), dtype=int)
    for i in range(nDerangements):
        permorder[i, :] = generate_derangement(n_splits * n_reps, n_splits)
    permorder = np.vstack((np.arange(n_splits * n_reps), permorder))


    # FFT cross-corr  # TODO look at np.copy(..) calls
    nn = T_steps
    R_EMEG = np.fft.rfft(norm(R_EMEG), n=nn, axis=-1)
    F_func = np.conj(np.fft.rfft(norm(np.copy(func)), n=nn, axis=-1))
    F_func = np.tile(F_func, (n_reps, 1))
    corrs = np.zeros((n_channels, nDerangements + 1, n_splits * n_reps, T_steps//2))

    for i, order in enumerate(permorder):
        deranged_EMEG = R_EMEG[:, order, :]
        corrs[:, i] = np.fft.irfft(deranged_EMEG * F_func)[:, :, :T_steps//2]

    if add_autocorr:
        auto_corrs = np.zeros((n_splits, T_steps//2))
        noise = norm(np.random.randn(func.shape[0], func.shape[1])) * 0
        noisy_func = norm(np.copy(func)) + noise
        nn = T_steps // 2 # T_steps

        F_noisy_func = np.fft.rfft(norm(noisy_func), n=nn, axis=-1)
        F_func = np.conj(np.fft.rfft(norm(func), n=nn, axis=-1))

        auto_corrs = np.fft.irfft(F_noisy_func * F_func)


    if verbose_timing: t2 = time.time(); print(f'Corr time: {round(t2-t1, 2)}s'); sys.stdout.flush()

    del F_func, deranged_EMEG, R_EMEG

    log_pvalues = ttest(corrs)

    if verbose_timing: t3 = time.time(); print(f'T-Test time: {round(t3-t2,2)}s\n'); sys.stdout.flush()

    latencies = np.linspace(start_lat, int(start_lat + (1000 * seconds_per_split)), 1 + T_steps//2)[:-1]
    if plot_name:
        plt.figure(1)
        maxs = np.max(-log_pvalues[:, :], axis=1)
        n_amaxs = 5
        amaxs = np.argpartition(maxs, -n_amaxs)[-n_amaxs:]
        amax = np.argmax(-log_pvalues) // (T_steps // 2)
        amaxs = [i for i in amaxs if i != amax] + [206]

        plt.plot(latencies, np.mean(corrs[amax, 0], axis=-2).T, 'r-', label=amax)
        plt.plot(latencies, np.mean(corrs[amaxs, 0], axis=-2).T, label=amaxs)
        std_null = np.mean(np.std(corrs[:, 1], axis=-2), axis=0).T * 3 / np.sqrt(n_reps * n_splits) # 3 pop std.s
        std_real = np.std(corrs[amax, 0], axis=-2).T * 3  / np.sqrt(n_reps * n_splits)
        av_real = np.mean(corrs[amax, 0], axis=-2).T
        #print(std_null)
        plt.fill_between(latencies, -std_null, std_null, alpha=0.5, color='grey')
        plt.fill_between(latencies, av_real - std_real, av_real + std_real, alpha=0.25, color='red')

        if add_autocorr:
            peak_lat_ind = np.argmax(-log_pvalues) % (T_steps // 2)
            peak_lat = latencies[peak_lat_ind]
            peak_corr = np.mean(corrs[amax, 0], axis=-2)[peak_lat_ind]
            print('peak lat, peak corr:', peak_lat, peak_corr)

            auto_corrs = np.mean(auto_corrs, axis=0)
            plt.plot(latencies, np.roll(auto_corrs, peak_lat_ind) * peak_corr / np.max(auto_corrs), 'k--', label='func auto-corr')

        
        #plt.plot(latencies, np.mean(corrs[0:300:15, 1], axis=-2).T, 'cyan')
        plt.axvline(0, color='k')
        plt.legend()
        plt.xlabel('latencies (ms)')
        plt.ylabel('Corr coef.')
        plt.savefig(f'{plot_name}_1.png')
        plt.clf()

        plt.figure(2)
        plt.plot(latencies, -log_pvalues[amax].T / np.log(10), 'r-', label=amax)
        plt.plot(latencies, -log_pvalues[amaxs].T / np.log(10), label=amaxs)
        plt.axvline(0, color='k')
        plt.legend()
        plt.xlabel('latencies (ms)')
        plt.ylabel('p-values')
        plt.savefig(f'{plot_name}_2.png')
        plt.clf()

    if save_pvalues_path:
        with h5py.File(save_pvalues_path + '.h5', 'w') as f:
            f.create_dataset('pvalues', data=pvalues)
            f.create_dataset('latencies', data=latencies)
            f.create_dataset('EMEG_names', data=EMEG_names)

    return np.max(-log_pvalues / np.log(10)), latencies[np.argmax(-log_pvalues) % (T_steps // 2)], np.argmax(-log_pvalues) // (T_steps // 2)


def latency_loop():
    participants = ['participant_01',
                    'participant_02',
                    'participant_03',
                    'participant_04',
                    'participant_05',
                    'pilot_01',
                    'pilot_02']
    test_path = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/scripts/'
    
    runs = [8, 8, 8, 6, 8, 4, 8]

    """lat_sig = np.zeros((len(participants), max(runs) + 1, 3)) # sig, lat, vert
    for p in range(len(participants)):
        for rep in range(runs[p]):
            print(participants[p], f'rep{rep}')
            sys.stdout.flush()
            #lat_sig[p, rep] = np.array(do_gridsearch(p=participants[p],
            #                                         rep=[f'_rep{r}' for r in range(0, rep + 1)],
            #                                         plot_name=None))
            lat_sig[p,rep] = np.array(do_gridsearch(func_name='neurogram_mr',
                                                    function_path='/Bruce_model/neurogramResults',
                                                    p=participants[p],
                                                    rep=f'_rep{rep}',
                                                    plot_name=None,
                                                    downsample_rate=1,
                                                    inverse_operator=None))
        lat_sig[p, -1] = np.array(do_gridsearch(func_name='neurogram_mr',
                                                function_path='/Bruce_model/neurogramResults',
                                                p=participants[p],
                                                rep='-ave',
                                                plot_name=None,
                                                downsample_rate=1,
                                                inverse_operator=None))
        

    np.save(f'{test_path}/lat_sig_bruce_0_10_corrs.npy', lat_sig)"""

    #lat_sig = np.load('lat_sig_3.npy')
    #lat_sig[0, 6, :] = np.array([357.761306, 80, 209])
    #lat_sig[0, 7, :] = np.array([409.303307, 80, 209])
    #np.save(f'{test_path}/lat_sig_3.npy', lat_sig)

    lat_sig = np.load('lat_sig.npy')

    """p_colors = ['r', 'g', 'b', 'y', 'k', 'cyan', 'pink']
    
    plt.figure(3)

    fig, ax = plt.subplots()

    log_correction = 1 #np.log(10)  # if logs are in base e then adjust
    for i in range(len(participants)):
        _lats = np.array([lat_sig[i,j,:] for j in range(lat_sig.shape[1] - 1) if lat_sig[i,j,0] != 0])
        ax.scatter(_lats[:, 1], _lats[:, 0] / log_correction, color=p_colors[i], marker='x')
        #ax.plot(_lats[:, 1], _lats[:, 0] / log_correction, color=p_colors[i], marker='x')
        ax.scatter(lat_sig[i, -1:, 1], lat_sig[i, -1:, 0] / log_correction, color=p_colors[i], marker='o', label=participants[i])
        #for j in range(_lats.shape[0]):
        #    ax.annotate(j+1, (_lats[j, 1], _lats[j, 0] / log_correction))"""

    #lats_new = [lat_sig[i, j, 1] for i in range(7) for j in range(8) if lat_sig[i, j, 1] < 110 and lat_sig[i,j,1] > 50]
    #print(lats_new)
    #print(np.std(lats_new))

    print(lat_sig[:, :, 1])

    """plt.xlabel('Latencies (ms)')
    plt.ylabel('-log(p)')
    plt.title('Indiv. Reps for Indivs (SPIKE TRAIN 0:10): Peak pvalue/latency')
    plt.legend()
    plt.xlim(32,200)
    plt.savefig(f'{test_path}/scat_example.png')"""


def hearing_corr_plot():
    lat_sig = np.load('lat_sig_2.npy')

    print([round(np.std(lat_sig[i, :runs[i], 1], ddof=1), 4) for i in range(7)])
    [print(list(lat_sig[i, :runs[i], 1])) for i in range(7)]
    stds_rem_ol = [1.753, 2.1339, 2.6095, 2.7325, 5.4292, 2.0817, 3.7733]
    
    plt.figure(3)
    hearing_results = [68, 74, 49, 64, 0, 68, 59]

    for i in [0,1,2,3,5,6]:
        plt.plot(lat_sig[i:i+1,-1,0] / np.log(10), hearing_results[i:i+1], 'o', color=p_colors[i], label=participants[i])
        # plt.plot(lat_sig[i,:,2], hearing_results[i:i+1] * 9, 'x', color=p_colors[i])
        

    plt.legend()
    plt.xlabel('-log(p)')
    plt.ylabel('hearing test (dB)')
    #plt.savefig(f'{test_path}/scat_example.png')
    
    hearing_results = [68, 74, 49, 64, 68, 59]
    print(stats.pearsonr(lat_sig[[0,1,2,3,5,6], -1, 0], hearing_results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gridsearch Params')  # TODO: fill this in or replace with something else
    parser.add_argument('--snr', type=float, default=3,
                        help='snr')
    args = parser.parse_args()

    participants = ['participant_01',
                    'participant_01b',
                    'participant_02',
                    'participant_03',
                    'participant_04',
                    'participant_05',
                    'pilot_01',
                    'pilot_02']

    reps = [f'_rep{i}' for i in range(8)]

    latency_loop()

    test_path = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/scripts/'

    """print(do_gridsearch(verbose_timing=True,
                        func_name='neurogram_ft',
                        function_path='/Bruce_model/neurogramResults',
                        p=participants[1],
                        rep='-ave',
                        ave_mode='ave',
                        plot_name=test_path+'example',
                        snr=args.snr,
                        downsample_rate=5,
                        inverse_operator=None))"""

    """print(do_gridsearch(verbose_timing=False,
                        func_name='d_IL2',
                        p=participants[1],
                        rep=rep,
                        ave_mode='ave',
                        plot_name=test_path+'example',
                        snr=args.snr,
                        downsample_rate=1,
                        inverse_operator=None))"""

    """rep0_rel_shifts = [0, -5, 11, 0, 3, -11, 49]
    ave_rel_shifts =  [81, 74, 92, 85, 78, 70, 136]

    min_shift = min(ave_rel_shifts)
    tshifts = [i-min_shift for i in ave_rel_shifts]
    print(tshifts); sys.stdout.flush()
    
    print(do_gridsearch(verbose_timing=True,
                        func_name='d_IL2',
                        p=participants[:-1],
                        rep='-ave',
                        ave_mode='ave',
                        plot_name=test_path+'example',
                        snr=args.snr,
                        downsample_rate=1,
                        inverse_operator=None,
                        p_tshift=tshifts,  ###############################
                        ))"""



    #f_root='/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/predicted_function_contours'
                  
    #bruce_neurons = (5,10)

    #bruce = load_function(f_root+'/Bruce_model/neurogramResults', 'neurogram_mr', 1, bruce_neurons)
    #d_STL = load_function(f_root+'/GMSloudness/stimulisig', 'IL2', 1)

    #plt.plot(norm(d_STL[:1000]), label='d_STL')
    #plt.plot(norm(bruce[:1000]), label=f'bruce ave: {bruce_neurons[0]}:{bruce_neurons[1]}')
    #plt.legend()
    #print(np.correlate(bruce[:5000], d_STL[:2000]))
    #plt.plot(np.correlate(norm(bruce[:5000]), norm(d_STL[:2000])))

    #plt.savefig('example_3.png')


