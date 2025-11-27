import numpy as np
import os
import matplotlib.pyplot as plt
import re
from matplotlib.colors import PowerNorm
from statistics import NormalDist
from kymata.io.nkg import load_expression_set
from tqdm import tqdm


def asr_models_loop_full():

    layer = 33 # 66 64 34
    neuron = 4096
    thres = 20 # 15
    x_upper = 800
    x_data = 'latency'

    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)
    thres = - np.log10(1 - ((1 - alpha)** (np.float128(1 / (200*370*(neuron*2+11)))))) # maybe we should get rid of the 2 here because we don't split the hemispheres

    plt.figure(3)
    fig, ax = plt.subplots()

    if x_data == 'latency':

        file_path = os.path.join('/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/en_all/all_tvl_gridsearch.nkg')
        expression_data_tvl = load_expression_set(file_path)

        for j in tqdm(range(layer)):
            expression_data_word = load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/size/salmonn_7b/fc2/expression_set/layer{j}/layer{j}_4095_gridsearch.nkg')
            for function in expression_data_word.functions:
                expression_data_word.rename({function: f'word_{function}'})
            expression_data_phone = load_expression_set(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/salmonn_7b_phone/fc2/expression_set/layer{j}/layer{j}_4095_gridsearch.nkg')
            for function in expression_data_phone.functions:
                expression_data_phone.rename({function: f'phone_{function}'})
            expression_data_layer = expression_data_word + expression_data_tvl + expression_data_phone

            data = expression_data_layer.best_functions()
            phone_list = []
            word_list = []
            tvl_list = []

            for i in range(data['value'].shape[0]):
                if -data['value'][i] > thres:
                    if 'phone' in data['function'][i]:
                        phone_list.append([data['latency'][i]*1000, -data['value'][i]])
                    elif 'word' in data['function'][i]:
                        word_list.append([data['latency'][i]*1000, -data['value'][i]])
                    else:
                        tvl_list.append([data['latency'][i]*1000, -data['value'][i]])

            phone_list = np.array(phone_list)
            word_list = np.array(word_list)
            tvl_list = np.array(tvl_list)
            try:
                scatter = ax.scatter(word_list[:, 0], np.ones((word_list.shape[0],))*j, c='red', marker='.', s=15, alpha=0.5, label = 'Salmonn word features')
            except:
                pass
            try:
                scatter = ax.scatter(phone_list[:, 0], np.ones((phone_list.shape[0],))*j, c='green', marker='.', s=15, alpha=0.5, label = 'Salmonn phonetic features')
            except:
                pass
            try:
                scatter = ax.scatter(tvl_list[:, 0], np.ones((tvl_list.shape[0],))*j, c='black', marker='.', s=15, alpha=0.2, label = 'TVL functions')
            except:
                pass

        # import ipdb;ipdb.set_trace()

        plt.xlabel('Latency (ms) relative to onset of the environment')


    ax.set_ylim(-1, layer)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=5)
    ax.axvline(x=0, color='k', linestyle='dotted')


    plt.ylabel('Salmonn layer number')

    # plt.title(f'Threshold -log(p-value): {thres}')
    plt.xlim(-200, x_upper)
    plt.savefig(f'/imaging/projects/cbu/kymata/analyses/tianyi/kymata-core/kymata-core-data/output/paper/scatter/salmonn_7b_phone_vs_word_v8_new_select', dpi=600, bbox_inches="tight")


if __name__ == '__main__':
    asr_models_loop_full()
    #latency_loop()