from transformers import AutoProcessor, Wav2Vec2Model
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import matplotlib.pyplot as plt
import whisper
import time
import os
import librosa
import argparse

def main():

  parser = argparse.ArgumentParser(description='Params')

  # Dataset specific
  parser.add_argument('--size', type=str, required=True)
  args = parser.parse_args()

  start_time = time.time()

  whisper_outs = True
  save_outs = True

  data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english_narratives'

  dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/audio/stimulus.wav', sr=16_000)

  T_max = 401 #seconds

  func_dir = '/imaging/woolgar/projects/Tianyi/data'
  # func_name = 'tiny'
  func_name = args.size

  features = {}
  timestamps = []
  text = []
  text_with_time = []


  if whisper_outs:
  # if True:

    dataset = dataset[:T_max*16_000]

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{func_name}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{func_name}")

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

  if whisper_outs and save_outs:

    if not os.path.isfile(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{func_name}/whisper_transcription.txt"):
      text = "\n".join(text)
      with open(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{func_name}/whisper_transcription.txt", "w") as file:
        file.write(text)
