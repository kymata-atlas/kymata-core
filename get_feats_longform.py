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

# data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
data_path = '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives'

# dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/audio/F00C_dataset3.wav', sr=16_000)

# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
# inputs = processor(dataset, sampling_rate=sampling_rate, return_tensors="pt")

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

# func_name = 'whisper_all_no_reshape'
# func_name = 'whisper_all_no_reshape_small_multi_timestamp'
func_name = 'ru_whisper_all_no_reshape_large_v2_longform'

# (512, 1284889)    3200 Hz
# (512, 642444) /2  1600
# (512, 321221) /2  800
# (512, 160610) /2  400
# (512, 80304) /2   200
# (512, 40152) /2   100 Hz
# (512, 20076) /2   20 Hz

# d_STL = load_function(f'{func_dir}/predicted_function_contours/GMSloudness/stimulisig',
#                       func_name='d_STL',
#                       bruce_neurons=(5, 10)
#                       )

# IL = load_function(f'{func_dir}/predicted_function_contours/GMSloudness/stimulisig',
#                     func_name='IL9',
#                     bruce_neurons=(5, 10)
#                     )

# func2 = load_function(f'{func_dir}/predicted_function_contours/asr_models/w2v_convs',
#                       func_name='conv_layer3',
#                       n_derivatives=0,
#                       n_hamming=0,
#                       nn_neuron=158, # 201, 158
#                       )

# func3 = load_function(f'{func_dir}/predicted_function_contours/Bruce_model/neurogramResults',
#                       func_name='neurogram_mr',
#                       n_derivatives=0,
#                       n_hamming=0,
#                       nn_neuron=158,
#                       bruce_neurons=(5, 10)
#                       )

# whisper_out = load_function_pre(f'{func_dir}/predicted_function_contours/asr_models/whisper_all',
#                       func_name='model.decoder.embed_tokens',
#                       )

# a = 300_000
# b = a + 1000

# for func in (d_STL, IL, func2, func3):
#   func.values /= np.max(func.values)
#   func.values /= np.sqrt(np.sum(func.values ** 2))


# func_a = IL
# func_b = func2 #d_STL + IL

# print(np.sum(func_a.values * func_b.values))

# plt.plot(func_a.values[a:b] / np.max(func_a.values[a:b]))
# plt.plot(func_b.values[a:b] / np.max(func_b.values[a:b]))
# plt.savefig('example_1.png')
features = {}
timestamps = []
text_with_time = []

def get_features(name):
  def hook(model, input, output):
    if isinstance(output,torch.Tensor) and (('model.decoder.layers' in name and 'final_layer_norm' in name) or 'proj_out' in name):
      if name in features.keys():
        if name == 'model.encoder.conv1' or name == 'model.encoder.conv2':
          # import ipdb;ipdb.set_trace()
          features[name] = torch.cat((features[name], output), -1)
        else:
          features[name] = torch.cat((features[name], output), -2)
      else:
        features[name] = output
  return hook

# def get_features(name):
#   def hook(model, input, output):
#       features[name] = output
#   return hook

########

if whisper_outs:
# if True:

  dataset = dataset[:T_max*16_000]

  processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
  model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
  # import ipdb;ipdb.set_trace()
  # for layer in model.children():
  #   layer.register_forward_hook(get_features("feats"))

  for name, layer in model.named_modules():
    # import ipdb;ipdb.set_trace()
    if isinstance(layer, torch.nn.Module):
        layer.register_forward_hook(get_features(name))

  inputs = processor(dataset, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=sampling_rate)
  # inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
  
  # generated_ids = model.generate(**inputs, return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
  # generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
  generated_ids = model.generate(**inputs, language='russian', return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
  # generated_ids = model.generate(**inputs, language='english', return_token_timestamps=False, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
  
  for i in range(len(generated_ids['segments'][0])):
    timestamps += generated_ids['segments'][0][i]['token_timestamps'].tolist()
  timestamps = np.array(timestamps)
  text = processor.batch_decode(**generated_ids, skip_special_tokens=False)[0]
  # transcription = processor.batch_decode(**generated_ids, skip_special_tokens=True)
  for i in range(generated_ids['sequences'].shape[1]):
    text_with_time.append(f'{processor.batch_decode(generated_ids["sequences"][:,i], skip_special_tokens=False)[0]}: {timestamps[i]}')

  # import ipdb;ipdb.set_trace()

  text_from_proj_out = processor.batch_decode(torch.argmax(features['proj_out'],dim=2)[0,:],skip_special_tokens=False)
  text_from_id = []
  for i in range(generated_ids['sequences'].shape[1]):
    text_from_id.append(processor.batch_decode(generated_ids["sequences"][:,i], skip_special_tokens=False)[0])
  indices_to_delete = [i for i, ele in enumerate(text_from_proj_out) if ele not in text_from_id]
  mask = torch.ones(features['proj_out'].size(1), dtype=torch.bool)
  mask[indices_to_delete] = False
  for key in features.keys():
    features[key] = features[key][:, mask, :]

  end_time = time.time()
  execution_time = end_time - start_time

  print(f"Execution time: {execution_time} seconds")
  # import ipdb;ipdb.set_trace()

else:

  features = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name}.npz')
  # features = np.load(f'{func_dir}/predicted_function_contours/asr_models/whisper_decoder.npz')

  # import ipdb;ipdb.set_trace()
  

if whisper_outs and save_outs:

  # s_num = T_max * 1000

  # import ipdb;ipdb.set_trace()

  # Check if the directory exists, if not, create it
  directory = f'{func_dir}/predicted_function_contours/asr_models/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Now save the data
  if not os.path.isfile(f'{directory}{func_name}.npz'):
    np.savez(f'{directory}{func_name}.npz', **features)
  if not os.path.isfile(f'{directory}{func_name}_timestamp.npy'):
    np.save(f'{directory}{func_name}_timestamp.npy', timestamps)
    plt.plot(timestamps)
    plt.savefig(f'kymata-toolbox-data/output/test/{func_name}_timestamp.png')
    plt.close()
  if not os.path.isfile(f"kymata-toolbox-data/output/test/{func_name}_transcription.txt"):
    with open(f"kymata-toolbox-data/output/test/{func_name}_transcription.txt", "w") as file:
      file.write(text)
  if not os.path.isfile(f"kymata-toolbox-data/output/test/{func_name}_transcription_time.txt"):
    text_with_time = "\n".join(text_with_time)
    with open(f"kymata-toolbox-data/output/test/{func_name}_transcription_time.txt", "w") as file:
      file.write(text_with_time)  

