import json
from itertools import cycle
from statistics import NormalDist
from typing import Tuple, Dict, List

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from matplotlib.lines import Line2D

from kymata.entities.expression import HexelExpressionSet, _FUNCTION, _LATENCY


class IPPMHexel(object):
    """
        Container to hold data about a hexel spike.
        
        Attributes
        ----------   
            function : the name of the function who caused the spike
            right_best_pairings : right hemisphere best pairings. pvalues are taken to the base 10 by default. latency is in milliseconds
            left_best_pairings : right hemisphere best pairings. same info as right for (latency, pvalue)
            description : optional written description
            github_commit : github commit of the function
    """
    
    def __init__(
                self, 
                function_name: str, 
                description: str=None, 
                github_commit: str=None,
            ): 
            self.function = function_name
            self.right_best_pairings = []
            self.left_best_pairings = []
            self.description = description
            self.github_commit = github_commit
            self.color = None

            self.input_stream = None
            
    def add_pairing(self, hemi: str, pairing: Tuple[float, float]):
        """
            Use this to add new pairings. Pair = (latency (ms), pvalue (log_10))

            Params
            ------
                hemi : leftHemisphere or rightHemisphere
                pairing : Corresponds to the best match to a hexel spike of form (latency (ms), pvalue (log_10))
        """
        if hemi == 'leftHemisphere':
            self.left_best_pairings.append(pairing)
        else:
            self.right_best_pairings.append(pairing)


def fetch_data(api: str) -> Dict[str, IPPMHexel]:
    """
        Fetches data from Kymata API and converts it into a dictionary of function names as keys
        and hexel objects as values. Advantage of dict is O(1) look-up and hexel object is readable
        access to attributes.
        
        Params
        ------
            api : URL of the API from which to fetch data
                
        Returns
        -------
            Dictionary containing data in the format [function name, hexel]
    """
    response = requests.get(api)
    resp_dict = json.loads(response.text)
    return build_hexel_dict_from_api_response(resp_dict)


def build_hexel_dict_from_expression_set(expression_set: HexelExpressionSet) -> Dict[str, IPPMHexel]:
    """
        Builds the dictionary from an ExpressionSet. This function builds a new dictionary
        which has function names (fast look-up) and only necessary data.

        Params
        ------
            dict_ : JSON dictionary of HTTP GET response object.

        Returns
        -------
            Dict of the format [function name, Hexel(func_name, id, left_pairings, right_pairings)]
    """
    best_functions_left, best_functions_right = expression_set.best_functions()
    hexels = {}
    for hemi in ['leftHemisphere', 'rightHemisphere']:
        best_functions = best_functions_left if hemi == "leftHemisphere" else best_functions_right
        for _idx, row in best_functions.iterrows():
            func = row[_FUNCTION]
            latency = row[_LATENCY] * 1000  # convert to ms
            pval = row["value"]
            if not func in hexels:
                hexels[func] = IPPMHexel(func)
            hexels[func].add_pairing(hemi, (latency, pval))
    return hexels


def build_hexel_dict_from_api_response(dict_: Dict) -> Dict[str, IPPMHexel]:
    """
        Builds the dictionary from response dictionary. Response dictionary has unneccesary 
        keys and does not have function names as keys. This function builds a new dictionary
        which has function names (fast look-up) and only necessary data.

        Params
        ------
            dict_ : JSON dictionary of HTTP GET response object.

        Returns
        -------
            Dict of the format [function name, Hexel(func_name, id, left_pairings, right_pairings)]
    """
    hexels = {}
    for hemi in ['leftHemisphere', 'rightHemisphere']:
        for (_, latency, pval, func) in dict_[hemi]:
            # we have id, latency (ms), pvalue (log_10), function name.
            # discard id as it conveys no useful information
            if not func in hexels:
                # first time seeing function, so create key and hexel object.
                hexels[func] = IPPMHexel(func)
            
            hexels[func].add_pairing(hemi, (latency, pow(10, pval)))
    
    return hexels


def stem_plot(
        hexels: Dict[str, IPPMHexel],
        title: str,
        timepoints: int=201,
        y_limit: float=pow(10, -100),
        number_of_hexels: int=200000,
        figheight: int=7,
        figwidth: int=12,
        ):
    """
        Plots a stem plot using hexels.

        Params
        ------
            hexels : Contains function spikes in the form of a Hexel object. All pairings are found there.
            title : Title of plot.
    """
    # estimate significance parameter
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)      # 5-sigma
    bonferroni_corrected_alpha = 1-(pow((1-alpha),(1/(2*timepoints*number_of_hexels))))

    # assign unique color to each function
    cycol = cycle(sns.color_palette("hls", len(hexels.keys())))
    for _, hexel in hexels.items():
        hexel.color = matplotlib.colors.to_hex(next(cycol))

    fig, (left_hem_expression_plot, right_hem_expression_plot) = plt.subplots(nrows=2, ncols=1, figsize=(figwidth, figheight))
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.84, left=0.08)

    custom_handles = []
    custom_labels = []
    for key, my_function in hexels.items():

        color = my_function.color
        label = my_function.function

        custom_handles.extend([Line2D([], [], marker='.', color=color, linestyle='None')])
        custom_labels.append(label)

        # left
        left = list(zip(*(my_function.left_best_pairings)))
        if len(left) != 0:
            x_left, y_left = left[0], left[1]
            left_color = np.where(np.array(y_left) <= bonferroni_corrected_alpha, color, 'black') # set all insignificant spikes to black
            left_hem_expression_plot.vlines(x=x_left, ymin=1, ymax=y_left, color=left_color)
            left_hem_expression_plot.scatter(x_left, y_left, color=left_color, s=20)

        # right
        right = list(zip(*(my_function.right_best_pairings)))
        if len(right) != 0:
            x_right, y_right = right[0], right[1]
            right_color = np.where(np.array(y_right) <= bonferroni_corrected_alpha, color, 'black') # set all insignificant spikes to black
            right_hem_expression_plot.vlines(x=x_right, ymin=1, ymax=y_right, color=right_color)
            right_hem_expression_plot.scatter(x_right, y_right, color=right_color, s=20)

    for plot in [right_hem_expression_plot, left_hem_expression_plot]:
        plot.set_yscale('log')
        plot.set_xlim(-200, 800)
        plot.set_ylim(1, y_limit)
        plot.axvline(x=0, color='k', linestyle='dotted')
        plot.axhline(y=bonferroni_corrected_alpha, color='k', linestyle='dotted')
        plot.text(-100, bonferroni_corrected_alpha, 'α*', bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.text(600, bonferroni_corrected_alpha, 'α*', bbox={'facecolor': 'white', 'edgecolor': 'none'}, verticalalignment='center')
        plot.set_yticks([1, pow(10,-50), pow(10,-100)])

    left_hem_expression_plot.set_title(title)
    left_hem_expression_plot.set_xticklabels([])
    right_hem_expression_plot.set_xlabel('Latency (ms) relative to onset of the environment')
    right_hem_expression_plot.xaxis.set_ticks(np.arange(-200, 800+1, 100))
    right_hem_expression_plot.invert_yaxis()
    left_hem_expression_plot.text(-180, y_limit * 10000000, 'left hemisphere', style='italic', verticalalignment='center')
    right_hem_expression_plot.text(-180, y_limit * 10000000, 'right hemisphere', style='italic', verticalalignment='center')
    y_axis_label = f'p-value (with α at 5-sigma, Bonferroni corrected)'
    left_hem_expression_plot.text(-275, 1, y_axis_label, verticalalignment='center',rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small', bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center', horizontalalignment='center', rotation='vertical')
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small', bbox_to_anchor=(1.2, 1))

def causality_violation_score(hexels: Dict[str, IPPMHexel], hierarchy: Dict[str, List[str]], hemi: str):
    assert(hemi == 'rightHemisphere'  or hemi == 'leftHemisphere')

    def get_latency(func_hexels: IPPMHexel, mini: bool):
        return (min(func_hexels.left_best_pairings, key=lambda x: x[0]) if hemi == 'leftHemisphere' else
                min(func_hexels.right_best_pairings, key=lambda x: x[0])) if mini else (
            max(func_hexels.left_best_pairings, key=lambda x: x[0]) if hemi == 'leftHemisphere' else
            max(func_hexels.right_best_pairings, key=lambda x: x[0]))

    causality_violations = 0
    total_arrows = 0
    for func, inc_edges in hierarchy.items():
        # arrows go from latest inc_edge spike to the earliest func spike
        child_latency = get_latency(hexels[func], mini=True)
        for inc_edge in inc_edges:
            parent_latency = get_latency(hexels[inc_edge], mini=False)
            if child_latency < parent_latency:
                causality_violations += 1
            total_arrows += 1

    return causality_violations / total_arrows if total_arrows != 0 else 0