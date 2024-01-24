import numpy as np
import os
import matplotlib.pyplot as plt


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

    #lat_sig = np.load('lat_sig_3.npy')
    #lat_sig[0, 6, :] = np.array([357.761306, 80, 209])
    #lat_sig[0, 7, :] = np.array([409.303307, 80, 209])
    #np.save(f'{test_path}/lat_sig_3.npy', lat_sig)

    lat_sig = np.zeros((len(participants), 9, 4)) # peak lat, peak corr, ind, -log(pval)

    log_dir = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/log_dIL2/'
    log_dir = '/imaging/projects/cbu/kymata/analyses/ollie/kymata-toolbox/log_dIL2_eeg/'

    file_list = os.listdir(log_dir)
    for i in range(len(file_list)):
        with open(log_dir + file_list[i], 'r') as f:
            a = f.readlines()
            if len(a):
                a = [float(j) for j in a[-2].split()[-4:]]  # a[-1]
                lat_sig[i % 19, i // 19, :] = a

    #print(lat_sig)

    plt.figure(3)
    fig, ax = plt.subplots()

    for i in range(len(participants)):
        if i == 6:
            continue
        _lats = np.array([lat_sig[i, j, :] for j in range(1, lat_sig.shape[1]) if lat_sig[i, j, 0] != 0])
        ax.scatter(_lats[:, 0], _lats[:, 3], marker='x', label=participants[i])
        # ax.scatter(lat_sig[i, :1, 0], lat_sig[i, :1, 3] / log_correction, marker='o')
        
        #for j in range(_lats.shape[0]):
        #    ax.annotate(j+1, (_lats[j, 1], _lats[j, 0] / log_correction))"""

    #lats_new = [lat_sig[i, j, 1] for i in range(7) for j in range(8) if lat_sig[i, j, 1] < 110 and lat_sig[i,j,1] > 50]
    #print(lats_new)
    #print(np.std(lats_new))

    plt.xlabel('Latencies (ms)')
    plt.ylabel('-log(p)')
    plt.title('Indiv. Reps for Indivs (d_IL2) (eeg only)')
    plt.legend()
    plt.xlim(60, 220)
    plt.savefig('scat_full__d_IL2_eeg.png')


if __name__ == '__main__':
    latency_loop()
