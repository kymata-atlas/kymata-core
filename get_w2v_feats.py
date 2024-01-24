
from transformers import AutoProcessor, Wav2Vec2Model
import torch
import matplotlib.pyplot as plt
import numpy as np

import librosa
# dataset, sampling_rate = librosa.load('/content/drive/MyDrive/Colab Notebooks/kymata/stimulus.wav', sr=16_000)

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(dataset, sampling_rate=sampling_rate, return_tensors="pt")

T_max = 410 #seconds

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

def model_convs(inputs):
  model_conv_outs = []
  with torch.no_grad():
    convs = model.feature_extractor.conv_layers
    inputs = inputs[:, None]
    for i, conv in enumerate(convs):
      inputs = conv(inputs)
      model_conv_outs.append(np.array(inputs[0]))
    return model_conv_outs

conv_outs = model_convs(inputs['input_values'][:, :16_000 * T_max])

# [print(i.shape) for i in conv_outs]

func_dict = {}

# TODO look at latency definitions and egdes
for i in range(len(conv_outs)):
    place_holder = np.zeros((conv_outs[i].shape[0], 410_000))
    for j in range(conv_outs[i].shape[0]):
        place_holder[j] = np.interp(np.linspace(0, 410, 410_001)[:-1], np.linspace(0, 410, conv_outs[i].shape[-1]), conv_outs[i][j])
    func_dict[f'conv_layer{i}'] = place_holder

np.savez(f'{data_path}/predicted_function_contours/asr_models/w2v_convs.npz', **func_dict)

