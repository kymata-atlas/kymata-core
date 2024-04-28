from transformers import AutoProcessor, Wav2Vec2Model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import whisper
import time
import os

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
func_name = 'whisper_all_no_reshape_large_v3_multi'

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

def get_features(name):
  def hook(model, input, output):
    if isinstance(output,torch.Tensor):
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

if whisper_outs and not os.path.isfile(f'{func_dir}/predicted_function_contours/asr_models/{func_name}.npz'):
# if True:

  dataset = dataset[:T_max*16_000]

  processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
  model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
  # import ipdb;ipdb.set_trace()
  # for layer in model.children():
  #   layer.register_forward_hook(get_features("feats"))

  for name, layer in model.named_modules():
    # import ipdb;ipdb.set_trace()
    if isinstance(layer, torch.nn.Module):
        layer.register_forward_hook(get_features(name))

  for i in range(14):
    if i == 13:
      segment = dataset[i*30*16_000:]
    else:
      segment = dataset[i*30*16_000:(i+1)*30*16_000]
    # inputs = processor(dataset, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=sampling_rate)
    inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
    
    # generated_ids = model.generate(**inputs, return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
    generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
    # import ipdb;ipdb.set_trace()
    # transcription = processor.batch_decode(**generated_ids, skip_special_tokens=True)

  end_time = time.time()
  execution_time = end_time - start_time
  print(f"Execution time: {execution_time} seconds")
  # import ipdb;ipdb.set_trace()

else:

  features = np.load(f'{func_dir}/predicted_function_contours/asr_models/{func_name}.npz')
  # features = np.load(f'{func_dir}/predicted_function_contours/asr_models/whisper_decoder.npz')

  # import ipdb;ipdb.set_trace()
  
########

if w2v_outs:
  model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
  #model.eval()
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

########

if wavlm_outs:
  from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

  wavlm_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus')

  def wavlm_convs(inputs):
    model_conv_outs = []
    with torch.no_grad():
      convs = wavlm_model.wavlm.feature_extractor.conv_layers
      inputs = inputs[:, None]
      for i, conv in enumerate(convs):
        inputs = conv(inputs)
        model_conv_outs.append(np.array(inputs[0]))
      return model_conv_outs

  conv_outs = wavlm_convs(inputs['input_values'][:, :16_000 * T_max])


########

if d2v_outs:
  from transformers import Data2VecAudioForCTC

  d2v_model = Data2VecAudioForCTC.from_pretrained("facebook/data2vec-audio-base")

  def d2v_convs(inputs):
    model_conv_outs = []
    with torch.no_grad():
      convs = d2v_model.data2vec_audio.feature_extractor.conv_layers
      inputs = inputs[:, None]
      for i, conv in enumerate(convs):
        inputs = conv(inputs)
        model_conv_outs.append(np.array(inputs[0]))
      return model_conv_outs

  conv_outs = d2v_convs(inputs['input_values'][:, :16_000 * T_max])


########

if hubert_outs:
  from transformers import HubertForCTC

  hubert_model = HubertForCTC.from_pretrained("facebook/hubert-base-ls960")

  def hubert_convs(inputs):
    model_conv_outs = []
    with torch.no_grad():
      convs = hubert_model.hubert.feature_extractor.conv_layers
      inputs = inputs[:, None]
      for i, conv in enumerate(convs):
        inputs = conv(inputs)
        model_conv_outs.append(np.array(inputs[0]))
      return model_conv_outs

  conv_outs = hubert_convs(inputs['input_values'][:, :16_000 * T_max])

########


if sum((w2v_outs, wavlm_outs, d2v_outs, hubert_outs)) and save_outs:
  [print(i.shape) for i in conv_outs]

  func_dict = {}

  s_num = T_max * 1000

  # TODO look at latency definitions and egdes
  for i in range(len(conv_outs)):
      place_holder = np.zeros((conv_outs[i].shape[0], s_num))
      for j in range(conv_outs[i].shape[0]):
          place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, T_max, conv_outs[i].shape[-1]), conv_outs[i][j])
      func_dict[f'conv_layer{i}'] = place_holder

# import ipdb;ipdb.set_trace()

if whisper_outs and save_outs:

  # s_num = T_max * 1000

  # Check if the directory exists, if not, create it
  directory = f'{func_dir}/predicted_function_contours/asr_models/'
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Now save the data
  np.savez(f'{directory}{func_name}.npz', **features)

  # np.savez(f'{func_dir}/predicted_function_contours/asr_models/whisper_all_no_reshape_large_v2.npz', **features)

  # func_dict = {}
  # for name,val in features.items():
  #   if 'decoder' in name or name == 'proj_out':
  #     print(name)
  #     if 'conv' in name or val.shape[0] != 1:
  #       place_holder = np.zeros((val.shape[1], s_num))
  #     else:
  #       place_holder = np.zeros((val.shape[2], s_num))
  #     for j in range(place_holder.shape[0]):
  #       if 'conv' in name:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[2]), val[0, j, :])
  #       elif val.shape[0] != 1:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[0]), val[:, j])
  #       else:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[1]), val[0, :, j])
  #     func_dict[name] = place_holder
  # np.savez(f'{data_path}/predicted_function_contours/asr_models/whisper_decoder.npz', **func_dict)

  # func_dict = {}
  # for name,val in features.items():
  #   if ('decoder' not in name) and ('encoder' in name):
  #     print(name)
  #     if 'conv' in name or val.shape[0] != 1:
  #       place_holder = np.zeros((val.shape[1], s_num))
  #     else:
  #       place_holder = np.zeros((val.shape[2], s_num))
  #     for j in range(place_holder.shape[0]):
  #       if 'conv' in name:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[2]), val[0, j, :])
  #       elif val.shape[0] != 1:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[0]), val[:, j])
  #       else:
  #         place_holder[j] = np.interp(np.linspace(0, T_max, s_num + 1)[:-1], np.linspace(0, 420, val.shape[1]), val[0, :, j])
  #     func_dict[name] = place_holder
  # np.savez(f'{data_path}/predicted_function_contours/asr_models/whisper_encoder.npz', **func_dict)
