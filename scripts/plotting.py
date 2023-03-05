import matplotlib.colors
from colorama import Fore
from colorama import Style
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import numpy as np
from itertools import cycle
import seaborn as sns

def plot_expression_plot():
    '''Generates an expression plot'''

    y_limit = pow(10, -100)
    alpha = pow(10, -13)

    print(f"{Fore.GREEN}{Style.BRIGHT}Filtering functions.{Style.RESET_ALL}")

    functions_to_include_in_model_selection = ['Combined Overall Loudness', 'CIECAM02-a']
    functions_to_plot = ['Combined Overall Loudness', 'CIECAM02-a']
    hexel_expression = {
        'combined-overall-loudness': {
                'name': 'Combined Overall Loudness',
                'description': 'Combines loudness after individual channel loudnesses',
                'github_commit': 'N/A',
                'left' : {
                    'latencies': np.arange(-200, 805, 5).tolist(),
                    'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80)]
                    },
                'right': {
                    'latencies': np.arange(-200, 800, 5).tolist(),
                    'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80),
                                lognuniform(low=0, high=1, size=201, base=-80)]
                }
        },
        'CIECAM02-a': {
            'name': 'CIECAM02-a',
            'description': 'Luminance dimension as defined by the CIECAM02 colour space',
            'github_commit': 'N/A',
            'left': {
                'latencies': np.arange(-200, 805, 5).tolist(),
                'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80)]
            },
            'right': {
                'latencies': np.arange(-200, 800, 5).tolist(),
                'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80)]
            }
        },
        'CIECAM02-b': {
            'name': 'CIECAM02-b',
            'description': 'yellow-blue dimension as defined by the CIECAM02 colour space',
            'github_commit': 'N/A',
            'left': {
                'latencies': np.arange(-200, 805, 5).tolist(),
                'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80)]
            },
            'right': {
                'latencies': np.arange(-200, 800, 5).tolist(),
                'pvalues': [lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80),
                            lognuniform(low=0, high=1, size=201, base=-80)]
            }
        }
    }

    print('Listing all available functions to plot:')
    for key, my_function in hexel_expression.items():
        print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")
    # test to confirm all xxx have the same number of hexels as pavlues

    hexel_expression_plotting = copy.deepcopy(hexel_expression)

    # filter to just the expression data we want in model selection
    # remove unwanted entries via requested_functions

    print('Listing all requested functions to include in model selection:')
    for key, my_function in hexel_expression_plotting.items():
        print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")

    print("...wrangling to retrieve only the pairing we are interested in.")
    for key, my_function in hexel_expression_plotting.items():
        for hemi in ['left','right']:
            my_function[hemi]['best_pairings'] = []
            for hexel in my_function[hemi]['pvalues']:
                best_latency = my_function[hemi]['latencies'][np.argmin(hexel)]
                best_pvalue = np.amin(hexel)
                my_function[hemi]['best_pairings'].append([best_latency, best_pvalue])
            print(my_function[hemi]['best_pairings'])

    #   for each hemisphere
    #       for each hexel
    #           go to xxx and find the best one.
    #           add to best_acrross_allmodels_pairins []

    # remove some from ploting
    # check these are subset of model selection!

    print('Listing all functions to plot:')
    for key, my_function in hexel_expression_plotting.items():
        print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")


    print("...adding colors.")
    cycol = cycle(sns.color_palette("Set1"))
    for key, my_function in hexel_expression_plotting.items():
          my_function['color'] = matplotlib.colors.to_hex(next(cycol))

    print(f"{Fore.GREEN}{Style.BRIGHT}Creating expression plots.{Style.RESET_ALL}")

    fig, (left_hem_expression_plot, right_hem_expression_plot) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    print(f"...print and color stems")

    custom_handles = []
    custom_labels = []
    for key, my_function in hexel_expression_plotting.items():

        color = my_function['color']
        label = my_function['name']

        custom_handles.extend([Line2D([], [], marker='.', color=color, linestyle='None')])
        custom_labels.append(label)

        # left
        x_left, y_left = zip(*my_function['left']['best_pairings'])
        left_color = np.where(np.array(y_left) <= alpha, color, 'black')
        left_hem_expression_plot.vlines(x=x_left, ymin=1, ymax=y_left, color=left_color)
        left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        x_right, y_right = zip(*my_function['right']['best_pairings'])
        right_color = np.where(np.array(y_right) <= alpha, color, 'black')
        right_hem_expression_plot.vlines(x=x_right, ymin=1, ymax=y_right, color=left_color)
        right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

    print(f"...formatting")

    # format shared axis qualities

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale('log')
        plot.set_xlim(-200, 800)
        plot.set_ylim(1, y_limit)
        plot.axvline(x=0, color='k', linestyle='dotted')
        plot.axhline(y=alpha, color='k', linestyle='dotted')
        plot.text(-100, alpha, 'α*', bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.text(600, alpha, 'α*', bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.set_yticks([1, pow(10,-50), pow(10,-100)])

    # format one-off axis qualities
    left_hem_expression_plot.set_title('Function Expression')
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel('Latency (ms) relative to onset of the environment')
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800+1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(-180, y_limit * 10000000, 'left hemisphere', style='italic', verticalalignment='center')
    right_hem_expression_plot.text(-180, y_limit * 10000000, 'right hemisphere', style='italic', verticalalignment='center')
    left_hem_expression_plot.text(-275, 1, 'p-value', verticalalignment='center',rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small', bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center', horizontalalignment='center', rotation='vertical')
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small', bbox_to_anchor=(1.2, 1))

    print(f"...saving")
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig('data/output-graphs/expression_plot.png')
    plt.show()
    plt.close()
    
    # 
    #for each function in xxx
    #    color based on function
    #xxx

    print(f"...saving expression plot")

def lognuniform(low=0, high=1, size=None, base=np.e):
        return np.random.uniform(low, high, size) / 100000000000
