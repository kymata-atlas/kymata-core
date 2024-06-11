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

test = False

w2v_outs, wavlm_outs, d2v_outs, hubert_outs = False, False, False, False
whisper_outs = True
save_outs = False

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
# data_path = '/imaging/projects/cbu/kymata/data/dataset_3-russian_narratives'

T_max = 401 #seconds

# func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
func_dir = '/imaging/woolgar/projects/Tianyi/data'

func_name = 'whisper_all_no_reshape_large_v3_teacher'

features = {}

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

def evaluate_whisper(audio_data, reference_text):
  inputs = processor(audio_data, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
  input_features = inputs['input_features']

  # Ensure inputs are padded to the model's expected input length (3000 frames)
  if input_features.size(-1) < 3000:
    input_features = torch.nn.functional.pad(input_features, (0, 3000 - input_features.size(-1)), mode='constant')
  elif input_features.size(-1) > 3000:
    input_features = input_features[:, :, :3000]

  # Encode audio features
  encoder_outputs = model.get_encoder()(input_features)

  # Tokenize reference text
  target_tokens = tokenizer(reference_text, return_tensors="pt").input_ids

  target_tokens = target_tokens[:, 2:-1]

  # Initialize predicted tokens
  predicted_tokens = []

  past_key_values= None

  # import ipdb;ipdb.set_trace()

  # Perform teacher forcing by feeding reference tokens to the decoder
  for t in range(target_tokens.size(1) + 1):

    # print(t)

    if t == 0:

      predicted_tokens.append(target_tokens[0][0].item())

    else:
      
      # decoder_input_ids = target_tokens[:, :t]
      decoder_input_ids = target_tokens[:, t - 1]
      
      outputs = model(
          input_features=input_features,
          decoder_input_ids=decoder_input_ids,
          # decoder_input_ids=torch.Tensor(predicted_tokens).int(),
          encoder_outputs=encoder_outputs,
          return_dict=True,
          past_key_values=past_key_values,
      )

      # import ipdb;ipdb.set_trace()
      
      logits = outputs.logits
      past_key_values = outputs.past_key_values
      predicted_token = torch.argmax(logits[:, -1, :], dim=-1).item()
      predicted_tokens.append(predicted_token)

  # Convert predicted tokens to text
  predicted_text = tokenizer.decode(predicted_tokens, skip_special_tokens=False)

  # Compute evaluation metrics (e.g., WER, CER, BLEU)

  # import ipdb;ipdb.set_trace()

  # Return evaluation results
  return predicted_text

if whisper_outs:
# if True:

  if test:
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
  else:
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

  for name, layer in model.named_modules():
    # import ipdb;ipdb.set_trace()
    if isinstance(layer, torch.nn.Module):
        layer.register_forward_hook(get_features(name))

  for i in range(14):
    audio_path = os.path.join(data_path, 'stimuli/tianyi_whisper', f'segment_{i}.wav')
    transcription_path = os.path.join(data_path, 'stimuli/tianyi_whisper', f'segment_{i}.txt')

    # Load audio segment
    audio_data, sr = librosa.load(audio_path, sr=16_000)

    # Read corresponding transcription
    with open(transcription_path, 'r') as file:
        reference_text = file.read()

    # reference_text = '<|startoftranscript|><|en|><|transcribe|><|notimestamps|> ' + reference_text + '<|endoftext|>'
    reference_text = '<|startoftranscript|><|en|><|transcribe|> ' + reference_text

    # Evaluate
    predicted_transcription = evaluate_whisper(audio_data, reference_text)

    # import ipdb;ipdb.set_trace()
  
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
