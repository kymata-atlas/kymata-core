import numpy as np
import os
import matplotlib.pyplot as plt
import re

def latency_loop():
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
                    'pilot_02'
                    ]
    part_map = {participants[i]:i for i in range(len(participants))}

    #lat_sig = np.load('lat_sig_3.npy')
    #lat_sig[0, 6, :] = np.array([357.761306, 80, 209])
    #lat_sig[0, 7, :] = np.array([409.303307, 80, 209])
    #np.save(f'{test_path}/lat_sig_3.npy', lat_sig)

    lat_sig = np.zeros((len(participants), 9, 4)) # peak lat, peak corr, ind, -log(pval)

    log_dir = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/log_dIL2/'
    # log_dir = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/log_dIL2_eeg/'

    rep_map = {f'rep{i}:':i+1 for i in range(8)}
    rep_map['ave:'] = 0

    file_list = os.listdir(log_dir)
    for i in range(len(file_list)):
        with open(log_dir + file_list[i], 'r') as f:
            a = f.readlines()
            if len(a):
                # print(a[-1][:20], participants[i % 19]) #, file_list[i])
                a = [j for j in a[-1].split()]  # a[-1]
                a0 = re.split('_|-', a[0])
                lat_sig[part_map['_'.join(a0[:2])], rep_map[a0[-1]], :] = [float(j) for j in a[-4:]]

    #print(lat_sig)

    plt.figure(3)
    fig, ax = plt.subplots()

    stds = []

    for i in range(len(participants)):
        if i == 6:
           continue
        _lats = np.array([lat_sig[i, j, :] for j in range(1, lat_sig.shape[1]) if lat_sig[i, j, 0] != 0])
        stds.append(np.std(_lats[:, 0]))
        # print(participants[i])
        # print([(j[0], j[1], j[3]) for j in _lats])
        ax.scatter(_lats[:, 2], _lats[:, 3], marker='x', label=participants[i])
        # ax.scatter(lat_sig[i, :1, 0], lat_sig[i, :1, 3], marker='o')

        #for j in range(_lats.shape[0]):
        #    ax.annotate(j+1, (_lats[j, 1], _lats[j, 0] / log_correction))"""

    #lats_new = [lat_sig[i, j, 1] for i in range(7) for j in range(8) if lat_sig[i, j, 1] < 110 and lat_sig[i,j,1] > 50]
    #print(lats_new)
    #print(np.std(lats_new))

    plt.xlabel('Latencies (ms)')
    plt.ylabel('-log(p)')
    plt.title('Indiv. Reps for Indivs (d_IL2)')
    # plt.legend()
    # plt.xlim(60, 220)
    plt.savefig('scat_full__d_IL2_testing.png')

    hearing_test_results = [
        68,   # part-01
        63.5, # part-01b
        74.5, # part-02
        49.5, # part-03
        64,   # part-04
        None, # part-05
        None, # part-07
        61,   # part-08 
        63.5, # part-09
        70,
        60.5,
        71,
        70.5,
        59.5,
        55,
        64.5,
        64,  # part-17
        68.5,  # pilot 01
        60,    # pilot 02
        ]
    
    plt.figure()

    plt.title('Hearing test pvals (d_IL2)')
    plt.xlabel('Hearing test result (dB)')
    plt.ylabel('kymata top pval')

    #print(stds)
    #print(hearing_test_results)

    for i in range(len(stds)):
        if stds[i] > 10:
            stds[i] = None

    plt.scatter(hearing_test_results[:], lat_sig[:, 0, 3])

    plt.savefig('scat_hearing.png')

    import scipy as sc

    sanned_pvals   = [lat_sig[i, 0, 3] for i in range(len(participants)) if hearing_test_results[i] is not None]
    sanned_hearing = [i for i in hearing_test_results if i is not None]

    print(sc.stats.pearsonr(sanned_hearing, sanned_pvals))




if __name__ == '__main__':
    latency_loop()
