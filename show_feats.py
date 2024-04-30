from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

func_dir = '/imaging/woolgar/projects/Tianyi/data'
func_name_1 = 'whisper_all_no_reshape_base_test'
func_name_2 = 'whisper_all_no_reshape_base_multi'

features_1 = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name_1}.npz')
features_2 = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name_2}.npz')

import ipdb;ipdb.set_trace()
plt.plot(features_1['model.encoder.conv1'][:, 177, :].reshape((84000)))
plt.savefig('test_1.png')
plt.close()
plt.plot(features_2['model.encoder.conv1'][:, 177, :].reshape((42000)))
plt.savefig('test_2.png')

