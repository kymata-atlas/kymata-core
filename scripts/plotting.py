import copy
from pathlib import Path
from itertools import cycle
from typing import Optional
from statistics import NormalDist

import numpy as np
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from colorama import Fore
from colorama import Style


def plot_expression_plot_script(save_to: Optional[Path] = None, verbose: bool = False):
    """Generates an expression plot"""

    y_limit = pow(10, -100)
    timepoints = 201
    number_of_hexels = 200000
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)  # 5-sigma
    bonferroni_corrected_alpha = 1 - (pow((1 - alpha), (1 / (2 * timepoints * number_of_hexels))))

    # TODO: there's probably a better way to do this switch with the logging module
    if verbose:
        print(f"{Fore.GREEN}{Style.BRIGHT}Filtering functions.{Style.RESET_ALL}")

    functions_to_include_in_model_selection = ['Combined Overall Loudness', 'CIECAM02-a', 'CIECAM02-b']
    functions_to_plot = ['Combined Overall Loudness', 'CIECAM02-a']
    hexel_expression = {
        'combined-overall-loudness': {
            'name': 'Combined Overall Loudness',
            'description': 'Combines loudness after individual channel loudnesses',
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

    if verbose:
        print('Listing all available functions to plot:')
        for key, my_function in hexel_expression.items():
            print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")

    # apply tests
    # test to confirm all xxx have the same number of hexels as pavlues
    # check to make sure all have save sized hexels

    hexel_expression_plotting = copy.deepcopy(hexel_expression)

    # filter to just the expression data we want in model selection
    # remove unwanted entries via requested_functions

    if verbose:
        print('Listing all requested functions to include in model selection:')
        for key, my_function in hexel_expression_plotting.items():
            print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")

    if verbose:
        print("...wrangling to retrieve only the pairing we are interested in.")

    for key, my_function in hexel_expression_plotting.items():
        for hemi in ['left', 'right']:
            my_function[hemi]['best_pairings'] = []
            for hexel in my_function[hemi]['pvalues']:
                best_latency = my_function[hemi]['latencies'][np.argmin(hexel)]
                best_pvalue = np.amin(hexel)
                my_function[hemi]['best_pairings'].append([best_latency, best_pvalue])
            if verbose:
                print(my_function[hemi]['best_pairings'])

    if verbose:
        print("...applying model selection.")
    #for hemi in ['left', 'right']:
        #for hexel in hemi.length():
            #[name-of-function] unzip find_name_of_best_function_at_hexel(hexel, hexel_expression_plotting)

    #    for each hexel in len hexels
    #       for key, my_function in hexel_expression_plotting.items():
    #           go to xxx and find the best one.
    #           add to best_acrross_allmodels_pairins []

    # remove some from ploting
    # check these are subset of model selection!

    if verbose:
        print('Listing all functions to plot:')
        for key, my_function in hexel_expression_plotting.items():
            print(f"{Fore.YELLOW}  {my_function['name']}{Style.RESET_ALL}")

    if verbose:
        print("...adding colors.")
    cycol = cycle(sns.color_palette("Set1"))
    for key, my_function in hexel_expression_plotting.items():
        my_function['color'] = matplotlib.colors.to_hex(next(cycol))

    if verbose:
        print(f"{Fore.GREEN}{Style.BRIGHT}Creating expression plots.{Style.RESET_ALL}")

    fig, (left_hem_expression_plot, right_hem_expression_plot) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    if verbose:
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
        left_color = np.where(np.array(y_left) <= bonferroni_corrected_alpha, color, 'black')
        left_hem_expression_plot.vlines(x=x_left, ymin=1, ymax=y_left, color=left_color)
        left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        x_right, y_right = zip(*my_function['right']['best_pairings'])
        right_color = np.where(np.array(y_right) <= bonferroni_corrected_alpha, color, 'black')
        right_hem_expression_plot.vlines(x=x_right, ymin=1, ymax=y_right, color=left_color)
        right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

    if verbose:
        print(f"...formatting")

    # format shared axis qualities

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale('log')
        plot.set_xlim(-200, 800)
        plot.set_ylim(1, y_limit)
        plot.axvline(x=0, color='k', linestyle='dotted')
        plot.axhline(y=bonferroni_corrected_alpha, color='k', linestyle='dotted')
        plot.text(-100, bonferroni_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.text(600, bonferroni_corrected_alpha, 'α*',
                  bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.set_yticks([1, pow(10, -50), pow(10, -100)])

    # format one-off axis qualities
    left_hem_expression_plot.set_title('Function Expression')
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel('Latency (ms) relative to onset of the environment')
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800 + 1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(-180, y_limit * 10000000, 'left hemisphere', style='italic',
                                  verticalalignment='center')
    right_hem_expression_plot.text(-180, y_limit * 10000000, 'right hemisphere', style='italic',
                                   verticalalignment='center')
    y_axis_label = f'p-value (with α at 5-sigma, Bonferroni corrected)'
    left_hem_expression_plot.text(-275, 1, y_axis_label, verticalalignment='center', rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small',
                                   bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center',
                                   horizontalalignment='center', rotation='vertical')
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small',
                                    bbox_to_anchor=(1.2, 1))

    if save_to is not None:
        if verbose:
            print(f"...saving")
        plt.rcParams['savefig.dpi'] = 300
        plt.savefig(Path(save_to))

    plt.show()
    plt.close()


def lognuniform(low=0, high=1, size=None, base=np.e):
    return np.random.uniform(low, high, size) / 1000000000000
