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

whisper_outs = True
save_outs = True
size = 'tiny'

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/audio/stimulus.wav', sr=16_000)

T_max = 401 #seconds

ablation_param = ['decoder', '2', 'test_1']
func_name = f'whisper_{size}_ablation_{ablation_param[0]}_{ablation_param[1]}_{str(ablation_param[2])}'
neurons_to_ablate = {f'model.{ablation_param[0]}.layers.{ablation_param[1]}.fc2': {range(0,300)}}
text = []

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

  processor = WhisperProcessor.from_pretrained(f"openai/whisper-{size}")
  model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{size}")

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
    
    generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
    text.append(processor.batch_decode(**generated_ids, skip_special_tokens=False)[0])

  end_time = time.time()
  execution_time = end_time - start_time
  print(f"Execution time: {execution_time} seconds")
  # import ipdb;ipdb.set_trace()


if whisper_outs and save_outs:

  # s_num = T_max * 1000

  # import ipdb;ipdb.set_trace()

  if not os.path.isfile(f"kymata-core-data/output/ablation/{func_name}_transcription.txt"):
    text = "\n".join(text)
    with open(f"kymata-core-data/output/ablation/{func_name}_transcription.txt", "w") as file:
      file.write(text)
