from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os

func_dir = '/imaging/woolgar/projects/Tianyi/data'
func_name_1 = 'whisper_all_no_reshape_base_ru'
func_name_2 = 'whisper_all_no_reshape_base_multi'

features_1 = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name_1}.npz')
features_2 = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name_2}.npz')

import ipdb;ipdb.set_trace()
plt.plot(features_1['model.encoder.conv1'][:, 177, :].reshape((42000)))
plt.plot(features_2['model.encoder.conv1'][:, 177, :].reshape((42000)))
plt.savefig('test.png')
