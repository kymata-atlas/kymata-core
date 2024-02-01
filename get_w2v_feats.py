
from transformers import AutoProcessor, Wav2Vec2Model
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

from kymata.io.functions import load_function

import librosa
# dataset, sampling_rate = librosa.load('/content/drive/MyDrive/Colab Notebooks/kymata/stimulus.wav', sr=16_000)

w2v_outs, wavlm_outs, d2v_outs, hubert_outs = False, False, False, False
save_outs = False

data_path = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'

dataset, sampling_rate = librosa.load(f'{data_path}/stimuli/stimulus.wav', sr=16_000)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

inputs = processor(dataset, sampling_rate=sampling_rate, return_tensors="pt")

T_max = 401 #seconds

func_dir = '/imaging/projects/cbu/kymata/data/dataset_4-english-narratives'

# (512, 1284889)    3200 Hz
# (512, 642444) /2  1600
# (512, 321221) /2  800
# (512, 160610) /2  400
# (512, 80304) /2   200
# (512, 40152) /2   100 Hz
# (512, 20076) /2   20 Hz

d_STL = load_function(f'{func_dir}/predicted_function_contours/GMSloudness/stimulisig',
                      func_name='d_STL',
                      bruce_neurons=(5, 10)
                      )

IL = load_function(f'{func_dir}/predicted_function_contours/GMSloudness/stimulisig',
                    func_name='IL9',
                    bruce_neurons=(5, 10)
                    )

func2 = load_function(f'{func_dir}/predicted_function_contours/asr_models/w2v_convs',
                      func_name='conv_layer3',
                      n_derivatives=0,
                      n_hamming=0,
                      nn_neuron=158, # 201, 158
                      )

func3 = load_function(f'{func_dir}/predicted_function_contours/Bruce_model/neurogramResults',
                      func_name='neurogram_mr',
                      n_derivatives=0,
                      n_hamming=0,
                      nn_neuron=158,
                      bruce_neurons=(5, 10)
                      )

a = 300_000
b = a + 1000

for func in (d_STL, IL, func2, func3):
  func.values /= np.max(func.values)
  func.values /= np.sqrt(np.sum(func.values ** 2))


func_a = IL
func_b = func2 #d_STL + IL

print(np.sum(func_a.values * func_b.values))

plt.plot(func_a.values[a:b] / np.max(func_a.values[a:b]))
plt.plot(func_b.values[a:b] / np.max(func_b.values[a:b]))
plt.savefig('example_1.png')

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

  np.savez(f'{data_path}/predicted_function_contours/asr_models/hubert_convs.npz', **func_dict)

