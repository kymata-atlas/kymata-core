from transformers import AutoProcessor, Wav2Vec2Model
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer
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

test = True

w2v_outs, wavlm_outs, d2v_outs, hubert_outs = False, False, False, False
whisper_outs = True
save_outs = True

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
# data_path = '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
# dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/audio/F00C_dataset3.wav', sr=16_000)

# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
# inputs = processor(dataset, sampling_rate=sampling_rate, return_tensors="pt")

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

func_name = 'whisper_all_no_reshape_large_v2_longform_teacher'

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


if whisper_outs:
# if True:

  dataset = dataset[:T_max*16_000]

  if test:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
  else:
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v2")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2")
  # import ipdb;ipdb.set_trace()
  # for layer in model.children():
  #   layer.register_forward_hook(get_features("feats"))

  for name, layer in model.named_modules():
    # import ipdb;ipdb.set_trace()
    if isinstance(layer, torch.nn.Module):
        layer.register_forward_hook(get_features(name))

  inputs = processor(dataset, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=sampling_rate)
  # inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
  with open('/imaging/woolgar/projects/Tianyi/kymata-toolbox/kymata-toolbox-data/output/test/transcription_en.txt', 'r') as file:
    file_content = file.read()
  labels = tokenizer(file_content, return_tensors="pt")

  import ipdb;ipdb.set_trace()

  model(input_features=inputs['input_features'], labels=labels['input_ids'])
  
  # for i in range(len(generated_ids['segments'][0])):
  #   timestamps += generated_ids['segments'][0][i]['token_timestamps'].tolist()
  # timestamps = np.array(timestamps)
  # text = processor.batch_decode(**generated_ids, skip_special_tokens=False)[0]
  # # transcription = processor.batch_decode(**generated_ids, skip_special_tokens=True)
  # for i in range(generated_ids['sequences'].shape[1]):
  #   text_with_time.append(f'{processor.batch_decode(generated_ids["sequences"][:,i], skip_special_tokens=False)[0]}: {timestamps[i]}')

  # # import ipdb;ipdb.set_trace()

  # text_from_proj_out = processor.batch_decode(torch.argmax(features['proj_out'],dim=2)[0,:],skip_special_tokens=False)
  # text_from_id = []
  # for i in range(generated_ids['sequences'].shape[1]):
  #   text_from_id.append(processor.batch_decode(generated_ids["sequences"][:,i], skip_special_tokens=False)[0])
  # indices_to_delete = [i for i, ele in enumerate(text_from_proj_out) if ele not in text_from_id]
  # mask = torch.ones(features['proj_out'].size(1), dtype=torch.bool)
  # mask[indices_to_delete] = False
  # for key in features.keys():
  #   features[key] = features[key][:, mask, :]

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
