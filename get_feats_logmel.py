from transformers import AutoProcessor, Wav2Vec2Model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import whisper
import time
import os

# import ipdb;ipdb.set_trace()

from kymata.io.functions import load_function, load_function_pre

import librosa
# dataset, sampling_rate = librosa.load('/content/drive/MyDrive/Colab Notebooks/kymata/stimulus.wav', sr=16_000)

start_time = time.time()

w2v_outs, wavlm_outs, d2v_outs, hubert_outs = False, False, False, False
whisper_outs = True
save_outs = True

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)

# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
# inputs = processor(dataset, sampling_rate=sampling_rate, return_tensors="pt")

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

# func_name = 'whisper_all_no_reshape'
# func_name = 'whisper_all_no_reshape_small_multi_timestamp'
func_name = 'whisper_all_no_reshape_base_multi_logmel'

logmel = []

# def get_features(name):
#   def hook(model, input, output):
#       features[name] = output
#   return hook

########

dataset = dataset[:T_max*16_000]

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")


for i in range(14):
  if i == 13:
    segment = dataset[i*30*16_000:]
  else:
    segment = dataset[i*30*16_000:(i+1)*30*16_000]
  # inputs = processor(dataset, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=sampling_rate)
  inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
  
  logmel.append(inputs['input_features'])

import ipdb;ipdb.set_trace()
end_time = time.time()
execution_time = end_time - start_time
# timestamps = np.concatenate(timestamps, axis = 1).reshape(-1)
print(f"Execution time: {execution_time} seconds")
# import ipdb;ipdb.set_trace()

