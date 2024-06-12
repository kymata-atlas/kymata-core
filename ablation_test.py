# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# import torch
# import numpy as np
# import time
# import librosa

# # Custom class to modify the Whisper model for neuron ablation
# class AblatedWhisperModel(WhisperForConditionalGeneration):
#     def __init__(self, config, neurons_to_ablate):
#         super().__init__(config)
#         self.neurons_to_ablate = neurons_to_ablate

#     def forward(self, *args, **kwargs):
#         import ipdb;ipdb.set_trace()
#         outputs = super().forward(*args, **kwargs)
#         if self.neurons_to_ablate:
#             # Ensure hidden states are present
#             hidden_states = outputs.decoder_hidden_states
#             if hidden_states is not None:
#                 for layer_index, neuron_indices in self.neurons_to_ablate.items():
#                     import ipdb;ipdb.set_trace()
#                     hidden_states[layer_index][:, :, neuron_indices] = 0
#             outputs.decoder_hidden_states = hidden_states
#         return outputs

# def ablation_study():
#     start_time = time.time()

#     whisper_outs = True

#     data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
#     dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
#     T_max = 401  # seconds

#     timestamps = []
#     text = []
#     text_with_time = []

#     if whisper_outs:
#         dataset = dataset[:T_max * 16_000]

#         processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
#         neurons_to_ablate = {'model.decoder.layers.15.final_layer_norm_1211': [1211]}  # Example: Ablate neurons 1 and 3 in the decoder
#         model = AblatedWhisperModel.from_pretrained("openai/whisper-tiny", neurons_to_ablate=neurons_to_ablate)

#         for i in range(14):
#             if i == 13:
#                 segment = dataset[i * 30 * 16_000:]
#             else:
#                 segment = dataset[i * 30 * 16_000:(i + 1) * 30 * 16_000]
#             inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")

#             generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True,
#                                            return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
#             timestamps.append(generated_ids['token_timestamps'].numpy()[:, 1:] + i * 30)
#             text.append(processor.batch_decode(generated_ids['sequences'], skip_special_tokens=False)[0])
#             for j in range(generated_ids['sequences'].shape[1]):
#                 text_with_time.append(f'{processor.batch_decode(generated_ids["sequences"][:, j], skip_special_tokens=False)[0]}: {generated_ids["token_timestamps"][:, j]}')

#         end_time = time.time()
#         execution_time = end_time - start_time
#         timestamps = np.concatenate(timestamps, axis=1).reshape(-1)
#         print(f"Execution time: {execution_time} seconds")
#         print(f"Transcriptions: {text}")
#         print(f"Timestamps: {timestamps}")

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import time
import librosa
import copy

# Function to apply ablation to specific neurons
def ablate_neurons(layer_output, neuron_indices):
    import ipdb;ipdb.set_trace()
    layer_output[:, :, neuron_indices] = 0
    return layer_output

# Custom function to hook into the transformer layers
def get_ablation_hook(neuron_indices):
    def hook(module, input, output):
        return ablate_neurons(output, neuron_indices)
    return hook

def ablation_study():
  start_time = time.time()

  whisper_outs = True
  test = True

  data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'
  dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
  T_max = 401  # seconds

  timestamps = []
  text = []
  text_with_time = []

  if whisper_outs:

    dataset = dataset[:T_max * 16_000]

    if test:
      processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
      model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    else:
      processor = WhisperProcessor.from_pretrained("openai/whisper-large")
      model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")

    # Neurons to ablate (example: ablate neurons 1 and 3 in the decoder's last layer)
    neurons_to_ablate = {'model.decoder.layers.0.final_layer_norm': [i for i in range(380)]}  # Modify indices based on your requirements

    # Register hooks for the specified layers and neurons
    for name, module in model.named_modules():
      if 'model.decoder.layers.15.final_layer_norm' in name:
        module.register_forward_hook(get_ablation_hook(neurons_to_ablate.get('model.decoder.layers.15.final_layer_norm', [])))

    for i in range(14):
      if i == 13:
        segment = dataset[i * 30 * 16_000:]
      else:
        segment = dataset[i * 30 * 16_000:(i + 1) * 30 * 16_000]
      inputs = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")

      generated_ids = model.generate(**inputs, language='english', return_token_timestamps=True,
                                      return_segments=True, return_dict_in_generate=True, num_segment_frames=480_000)
      timestamps.append(generated_ids['token_timestamps'].numpy()[:, 1:] + i * 30)
      text.append(processor.batch_decode(generated_ids['sequences'], skip_special_tokens=False)[0])
      for j in range(generated_ids['sequences'].shape[1]):
        text_with_time.append(f'{processor.batch_decode(generated_ids["sequences"][:, j], skip_special_tokens=False)[0]}: {generated_ids["token_timestamps"][:, j]}')

    end_time = time.time()
    execution_time = end_time - start_time
    timestamps = np.concatenate(timestamps, axis=1).reshape(-1)
    print(f"Execution time: {execution_time} seconds")
    print(f"Transcriptions: {text}")
    print(f"Timestamps: {timestamps}")

  import ipdb;ipdb.set_trace()

ablation_study()
