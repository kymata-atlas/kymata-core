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
  func_name = args.size

  text = []


  if whisper_outs:
  # if True:

    dataset = dataset[:T_max*16_000]

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{func_name}")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{func_name}")

    inputs = processor(dataset, return_tensors="pt", truncation=False, padding="longest", return_attention_mask=True, sampling_rate=sampling_rate)
    generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True, return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
    text = processor.batch_decode(**generated_ids, skip_special_tokens=False)[0]

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

  if whisper_outs and save_outs:
    if not os.path.isfile(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{func_name}/whisper_transcription_longform.txt"):
      with open(f"/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/{func_name}/whisper_transcription_longform.txt", "w") as file:
        file.write(text)

if __name__ == '__main__':
  main()