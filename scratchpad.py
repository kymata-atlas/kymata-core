import numpy as np
import h5py
import matplotlib.pyplot as plt

from kymata.io.mne import load_emeg_pack


func_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/predicted_function_contours/GMSloudness/stimulisig.h5'

participants = ['participant_01',
                'participant_01b',
                'participant_02',
                'participant_03',
                'participant_04',
                'participant_05',
                'pilot_01',
                'pilot_02']

emeg_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives/intrim_preprocessing_files/3_trialwise_sensorspace/evoked_data/'

channel = 206

emeg_dict = {}

lat = 0.082

shift_per_second = 1.000_537_5

emeg_paths = [emeg_path + i + '-ave' for i in participants[:]]
emeg, ch_names = load_emeg_pack(emeg_paths,
                                need_names=False,
                                ave_mode='ave',
                                inverse_operator=None, #inverse_operator, # set to None/inverse_operator if you want to run on sensor space/source space
                                p_tshift=None,
                                snr=1)
emeg_dict[f'all_participants_ave-chan{channel}'] = np.interp(np.linspace(lat*shift_per_second, (400+lat)*shift_per_second, 400_001)[:-1], np.linspace(-0.2, 401.8, 402001), emeg[channel, 0])


for i in range(8):
    emeg_paths = [emeg_path + i + '-ave' for i in participants[i:i+1]]
    emeg, ch_names = load_emeg_pack(emeg_paths,
                                    need_names=False,
                                    ave_mode='ave',
                                    inverse_operator=None, #inverse_operator, # set to None/inverse_operator if you want to run on sensor space/source space
                                    p_tshift=None,
                                    snr=1)
    #print(emeg.shape)
    emeg_dict[participants[i]+f'-chan{channel}-ave'] = np.interp(np.linspace(lat*shift_per_second, (400+lat)*shift_per_second, 400_001)[:-1], np.linspace(-0.2, 401.8, 402001), emeg[channel, 0])
    #print(emeg_dict[participants[i]].shape, '--')

for channel in [206, 209]:
    for i in range(8):
        emeg_paths = [emeg_path + f'participant_01_rep{i}' ]
        emeg, ch_names = load_emeg_pack(emeg_paths,
                                        need_names=False,
                                        ave_mode='ave',
                                        inverse_operator=None, #inverse_operator, # set to None/inverse_operator if you want to run on sensor space/source space
                                        p_tshift=None,
                                        snr=1)
        emeg_dict[f'participant_01-chan{channel}_rep{i}'] = np.interp(np.linspace(lat*shift_per_second, (400+lat)*shift_per_second, 400_001)[:-1], np.linspace(-0.2, 401.8, 402001), emeg[channel, 0])


with h5py.File(func_path, "r") as f:
    func_dict = {}
    for key in f.keys():
        if 'dd' in key or key in ('d_IL', 'd_STL', 'd_LTL'):
            continue
        if key in ('IL', 'STL', 'LTL'):
            func_dict[key] = np.array(f[key]).T.flatten()
        else:
            if 'comb' in key:
                func_dict[key] = np.array(f[key])
            else:
                func_dict[key] = np.array(f[key]).flatten()


np.savez('dataset4_TVL2020.npz', **func_dict)
np.savez('dataset4_EMEG_latency82.npz', **emeg_dict)

"""for i in emeg_dict:
    print(i, np.corrcoef(emeg_dict[i], func_dict['d_IL2'])[0, 1])

plt.plot(emeg_dict['participant_01-chan206-ave'][:2000])
plt.plot(func_dict['d_IL2'][:2000])
plt.savefig('example_2.png')"""

