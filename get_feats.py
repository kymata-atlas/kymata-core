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

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

# func_name = 'whisper_all_no_reshape'
# func_name = 'whisper_all_no_reshape_small_multi_timestamp'
func_name = 'whisper_all_no_reshape_large'

features = {}
timestamps = []
text = []
text_with_time = []

def get_features(name):
  def hook(model, input, output):
    # if isinstance(output,torch.Tensor) and (('model.decoder.layers' in name and 'final_layer_norm' in name) or 'proj_out' in name):
    # if isinstance(output,torch.Tensor) and (('model.decoder.layers' in name) or 'proj_out' in name):
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

if whisper_outs:
# if True:

  dataset = dataset[:T_max*16_000]

  processor = WhisperProcessor.from_pretrained("openai/whisper-large")
  model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
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
    # generated_ids = model.generate(**inputs, language='english', return_token_timestamps=False, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
    # import ipdb;ipdb.set_trace()
    timestamps.append(generated_ids['token_timestamps'].numpy()[:, 1:] + i * 30)
    text.append(processor.batch_decode(**generated_ids, skip_special_tokens=False)[0])
    for i in range(generated_ids['sequences'].shape[1]):
      text_with_time.append(f'{processor.batch_decode(generated_ids["sequences"][:,i], skip_special_tokens=False)[0]}: {generated_ids["token_timestamps"][:,i]}')
    # transcription = processor.batch_decode(**generated_ids, skip_special_tokens=True)

  end_time = time.time()
  execution_time = end_time - start_time
  timestamps = np.concatenate(timestamps, axis = 1).reshape(-1)
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
  if not os.path.isfile(f"{directory}{func_name}_whisper_transcription.txt"):
    text = "\n".join(text)
    with open(f"{directory}{func_name}_whisper_transcription.txt", "w") as file:
      file.write(text)
  if not os.path.isfile(f"kymata-toolbox-data/output/test/{func_name}_transcription_time.txt"):
    text_with_time = "\n".join(text_with_time)
    with open(f"kymata-toolbox-data/output/test/{func_name}_transcription_time.txt", "w") as file:
      file.write(text_with_time)  

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
