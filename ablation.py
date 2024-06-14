from transformers import AutoProcessor, Wav2Vec2Model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import whisper
import time
import os

# import ipdb;ipdb.set_trace()

import librosa
# dataset, sampling_rate = librosa.load('/content/drive/MyDrive/Colab Notebooks/kymata/stimulus.wav', sr=16_000)

start_time = time.time()

w2v_outs, wavlm_outs, d2v_outs, hubert_outs = False, False, False, False
whisper_outs = True
save_outs = True
test = True

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

# func_name = 'whisper_all_no_reshape'
# func_name = 'whisper_all_no_reshape_small_multi_timestamp'
ablation_param = ['decoder', '2', 'test_1']
func_name = f'whisper_tiny_ablation_{ablation_param[0]}_{ablation_param[1]}_{str(ablation_param[2])}'

# neurons_to_ablate = {f'model.{ablation_param[0]}.layers.{ablation_param[1]}.final_layer_norm': [int(ablation_param[2])]}
# neurons_to_ablate = {f'model.{ablation_param[0]}.layers.{ablation_param[1]}.final_layer_norm': {range(5,1000)}}
neurons_to_ablate = {f'model.{ablation_param[0]}.layers.{ablation_param[1]}.final_layer_norm': {range(0,300)}}
timestamps = []
text = []
text_with_time = []

def ablation(name, neuron):
  def hook(model, input, output):
    if isinstance(output,torch.Tensor):
      # import ipdb;ipdb.set_trace()
      for i in neuron[name]:
        output[:, :, i] = 0
    return output
  return hook

if whisper_outs:
# if True:

  dataset = dataset[:T_max*16_000]

  if test:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
  else:
    processor = WhisperProcessor.from_pretrained("openai/whisper-large")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
  # import ipdb;ipdb.set_trace()
  # for layer in model.children():
  #   layer.register_forward_hook(get_features("feats"))

  for name, layer in model.named_modules():
    # import ipdb;ipdb.set_trace()
    if isinstance(layer, torch.nn.Module) and name in neurons_to_ablate.keys():
        layer.register_forward_hook(ablation(name, neurons_to_ablate))

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
    # import ipdb;ipdb.set_trace()

  end_time = time.time()
  execution_time = end_time - start_time
  timestamps = np.concatenate(timestamps, axis = 1).reshape(-1)
  print(f"Execution time: {execution_time} seconds")
  # import ipdb;ipdb.set_trace()


if whisper_outs and save_outs:

  # s_num = T_max * 1000

  # import ipdb;ipdb.set_trace()

  directory = f'{func_dir}/predicted_function_contours/asr_models/'
  if not os.path.isfile(f'{directory}{func_name}_timestamp.npy'):
    plt.plot(timestamps)
    plt.savefig(f'kymata-toolbox-data/output/test/{func_name}_timestamp.png')
    plt.close()
  if not os.path.isfile(f"kymata-toolbox-data/output/test/{func_name}_transcription.txt"):
    text = "\n".join(text)
    with open(f"kymata-toolbox-data/output/test/{func_name}_transcription.txt", "w") as file:
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
