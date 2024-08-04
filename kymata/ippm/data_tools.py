import json
from itertools import cycle
from statistics import NormalDist
from typing import Tuple, Dict, List
from collections import namedtuple

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import requests
import seaborn as sns
from matplotlib.lines import Line2D
import math
from kymata.entities.expression import HexelExpressionSet, DIM_FUNCTION, DIM_LATENCY

Node = namedtuple('Node', 'magnitude position inc_edges')


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
            func = row[DIM_FUNCTION]
            latency = row[DIM_LATENCY] * 1000  # convert to ms
            pval = row["value"]
            if func not in hexels:
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
            if func not in hexels:
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
    y_axis_label = 'p-value (with α at 5-sigma, Bonferroni corrected)'
    left_hem_expression_plot.text(-275, 1, y_axis_label, verticalalignment='center',rotation='vertical')
    right_hem_expression_plot.text(0, 1, '   onset of environment   ', color='white', fontsize='x-small', bbox={'facecolor': 'grey', 'edgecolor': 'none'}, verticalalignment='center', horizontalalignment='center', rotation='vertical')
    left_hem_expression_plot.legend(handles=custom_handles, labels=custom_labels, fontsize='x-small', bbox_to_anchor=(1.2, 1))

    plt.show()


def causality_violation_score(denoised_hexels: Dict[str, IPPMHexel], hierarchy: Dict[str, List[str]], hemi: str, inputs: List[str]) -> Tuple[float, int, int]:
    """
        Assumption: hexels are denoised. Otherwise, it doesn't really make sense to check the min/max latency of noisy hexels.

        A score calculated on denoised hexels that calculates the proportion of arrows in IPPM that are going backward in time.
        It assumes that the function hierarchy is correct, which may not always be correct, so you must use it with caution. 

        Algorithm
        ----------
        violations = 0
        total_arrows = 0
        for each func_name, parents_list in hierarchy:
            child_lat = min(hexels[func])
            for parent in parents_list:
                parent_lat = max(hexels[parent])
                if child_lat < parent_lat:
                    violations++
                total_arrows++
        return violations / total_arrows if total_arrows > 0 else 0
    """
    
    assert(hemi == 'rightHemisphere'  or hemi == 'leftHemisphere')

    def get_latency(func_hexels: IPPMHexel, mini: bool):
        return (min(func_hexels.left_best_pairings, key=lambda x: x[0]) if hemi == 'leftHemisphere' else
                min(func_hexels.right_best_pairings, key=lambda x: x[0])) if mini else (
            max(func_hexels.left_best_pairings, key=lambda x: x[0]) if hemi == 'leftHemisphere' else
            max(func_hexels.right_best_pairings, key=lambda x: x[0]))

    causality_violations = 0
    total_arrows = 0
    for func, inc_edges in hierarchy.items():
        # essentially: if max(parent_spikes_latency) > min(child_spikes_latency), there will be a backwards arrow in time.
        # arrows go from latest inc_edge spike to the earliest func spike

        if func in inputs:
            continue

        if hemi == 'leftHemisphere':
            if len(denoised_hexels[func].left_best_pairings) == 0:
                continue
        else:
            if len(denoised_hexels[func].right_best_pairings) == 0:
                continue
        
        child_latency = get_latency(denoised_hexels[func], mini=True)[0]
        for inc_edge in inc_edges:
            if inc_edge in inputs:
                # input node, so parent latency is 0
                parent_latency = 0
                if child_latency < parent_latency:
                    causality_violations += 1
                total_arrows += 1
                continue

            # We need to ensure the function has significant spikes
            if hemi == 'leftHemisphere':
                if len(denoised_hexels[inc_edge].left_best_pairings) == 0:
                    continue
            else:
                if len(denoised_hexels[inc_edge].right_best_pairings) == 0:
                    continue
                    
            parent_latency = get_latency(denoised_hexels[inc_edge], mini=False)[0]
            if child_latency < parent_latency:
                causality_violations += 1
            total_arrows += 1

    return (
        causality_violations / total_arrows if total_arrows != 0 else 0,
        causality_violations,
        total_arrows)

def function_recall(noisy_hexels: Dict[str, IPPMHexel], funcs: List[str], ippm_dict: Dict[str, Node], hemi: str) -> Tuple[float]:
    """
        This is the second scoring metric: function recall. It illustrates what proportion out of functions in the noisy hexels are detected as part of IPPM. E.g., 9 functions but only 8 found => 8/9 = function recall. Use this along with causality violation to evaluate IPPMs and analyse their strengths and weaknesses. 
        
        One thing to note is that the recall depends upon the nature of the dataset. If certain functions have no significant spikes, there is an inherent bias present in the dataset. We can never get the function recall to be perfect no matter what algorithm we employ. Therefore, the function recall is based on what we can actually do with a dataset. E.g., 9 functions in the hierarchy but in the noisy hexels we find only 7 of the 9 functions. Moreover, after denoising we find that there are only 6 functions in the hierarchy. The recall will be 6/7 rather than 6/9 since there were only 7 to be found to begin with.

        Params
        ------
        hexels: the noisy hexels that we denoise and feed into IPPMBuilder. It must be the same dataset.
        funcs: list of functions that are in our hierarchy. Don't include the input function, e.g., input_cochlear.
        ippm_dict: the return value from IPPMBuilder. It contains node names as keys and Node objects as values.
        hemi: leftHemisphere or rightHemisphere

        Returns
        -------
        A ratio indicating how many channels were incorporated into the IPPM out of all relevant channels.
    """
    assert(hemi == 'rightHemisphere' or hemi == 'leftHemisphere')

    # Step 1: Calculate significance level
    alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)      
    bonferroni_corrected_alpha = 1-(pow((1-alpha),(1/(2*201*200000))))
    funcs_present_in_data = 0
    detected_funcs = 0
    for func in funcs:
        pairings = noisy_hexels[func].right_best_pairings if hemi == 'rightHemisphere' else noisy_hexels[func].left_best_pairings
        for latency, spike in pairings:
            # Step 2: Find a pairing that is significant
            if spike <= bonferroni_corrected_alpha:
                funcs_present_in_data += 1
                
                # Step 3: Found a function, look in ippm_dict.keys() for the function.
                for node_name in ippm_dict.keys():
                    if func in node_name:
                        # Step 4: If found, then increment detected_funcs. Also increment funcs_pressent
                        detected_funcs += 1
                        break
                break

    # Step 3: Return [ratio, numerator, denominator] primarily because both the denominator and numerator can vary.
    return (detected_funcs / funcs_present_in_data if funcs_present_in_data > 0 else 0, 
            detected_funcs, 
            funcs_present_in_data)
    

def convert_to_power10(hexels: Dict[str, IPPMHexel]) -> Dict[str, IPPMHexel]:
    """
        Utility function to take data from the .nkg format and convert it to power of 10, so it can be used for IPPMs.

        Parameters
        ------------
        hexels: dict function_name as key and hexel object as value. Hexels contain pairings for left/right.
        
        Returns
        --------
        same dict but the pairings are all raised to power x. E.g., pairings = [(lat1, x), ..., (latn, xn)] -> [(lat1, 10^x), ..., (latn, 10^xn)]
    """
    for func, hexel in hexels.items():
        hexels[func].right_best_pairings = list(map(lambda x: (x[0], math.pow(10, x[1])), hexels[func].right_best_pairings))
        hexels[func].left_best_pairings = list(map(lambda x: (x[0], math.pow(10, x[1])), hexels[func].left_best_pairings))
    return hexels


def remove_excess_funcs(to_retain: List[str], hexels: Dict[str, IPPMHexel]) -> Dict[str, IPPMHexel]:
    """
        Utility function to distill the hexels down to a subset of functions. Use this to visualise a subset of functions for time-series.
        E.g., you want the time-series for one function, so just pass it wrapped in a list as to_retain

        Parameters
        ----------
        to_retain: list of functions we want to retain in the hexels dict
        hexels: hexels: dict function_name as key and hexel object as value. Hexels contain pairings for left/right.

        Returns
        -------
        hexels but all functions that aren't in to_retain are filtered.
    """

    funcs = list(hexels.keys()) # need this because we can't remove from the dict while also iterating over it.
    for func in funcs:
        if func not in to_retain:
            # delete
            hexels.pop(func)
    return hexels


def plot_k_dist_1D(pairings: List[Tuple[float, float]], k: int=4, normalise: bool=False):
    """
        This could be optimised further but since we aren't using it, we can leave it as it is.
            
        A utility function to plot the k-dist graph for a set of pairings. Essentially, the k dist graph plots the distance
        to the kth neighbour for each point. By inspecting the gradient of the graph, we can gain some intuition behind the density of 
        points within the dataset, which can feed into selecting the optimal DBSCAN hyperparameters.

        For more details refer to section 4.2 in https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf

        Parameters
        ----------
        pairings: list of pairings extracted from a hexel. It contains the pairings for one function and one hemisphere
        k: the k we use to find the kth neighbour. Paper above advises to use k=4.
        normalise: whether to normalise before plotting the k-dist. It is important because the k-dist then equally weights both dimensions.

        Returns
        -------
        Nothing but plots a graph.
    """
    
    alpha = 3.55e-15
    X = pd.DataFrame(columns=['Latency'])
    for latency, spike in pairings:
        if spike <= alpha:
            X.loc[len(X)] = [latency]

    if normalise:
        X = normalize(X)
            
    distance_M = euclidean_distances(X) # rows are points, columns are other points same order with values as distances
    k_dists = []
    for r in range(len(distance_M)):
        sorted_dists = sorted(distance_M[r], reverse=True) # descending order
        k_dists.append(sorted_dists[k]) # store k-dist
    sorted_k_dists = sorted(k_dists, reverse=True)
    plt.plot(list(range(0, len(sorted_k_dists))), sorted_k_dists)
    plt.show()


def copy_hemisphere(
    hexels_to: Dict[str, IPPMHexel], 
    hexels_from: Dict[str, IPPMHexel], 
    hemi_to: str,
    hemi_from: str,
    func: str = None):
    """
        Utility function to copy a hemisphere onto another one. The primary use-case is to plot the denoised hemisphere against the
        noisy hemisphere using the same hexel object. I.e., copy right hemisphere to left; denoise on right; plot right vs left.

        Parameters
        ----------
        hexels_to: Hexels we are writing into. Could be (de)noisy hexels.
        hexels_from: Hexels we are copying from
        hemi_to: the hemisphere we index into when we write into hexels_to. E.g., hexels_to[hemi_to] = hexels_from[hemi_from]
        hemi_from: the hemisphere we index into when we copy the hexels from hexels_from. 
        func: if func != None, we only copy one function. Otherwise, we copy all.
        
        Returns
        -------
        Nothing, everything is done in-place. I.e., hexels_to is now updated.
    """
    if func:
        # copy only one function
        if hemi_to == 'rightHemisphere' and hemi_from == 'rightHemisphere':
            hexels_to[func].right_best_pairings = hexels_from[func].right_best_pairings
        elif hemi_to == 'rightHemisphere' and hemi_from == 'leftHemisphere':
            hexels_to[func].right_best_pairings = hexels_from[func].left_best_pairings
        elif hemi_to == 'leftHemisphere' and hemi_from == 'rightHemisphere':
            hexels_to[func].left_best_pairings = hexels_from[func].right_best_pairings
        else:
            hexels_to[func].left_best_pairings = hexels_from[func].left_best_pairings
        return
        
    for func, hexel in hexels_from.items():
        if hemi_to == 'rightHemisphere' and hemi_from == 'rightHemisphere':
            hexels_to[func].right_best_pairings = hexels_from[func].right_best_pairings
        elif hemi_to == 'rightHemisphere' and hemi_from == 'leftHemisphere':
            hexels_to[func].right_best_pairings = hexels_from[func].left_best_pairings
        elif hemi_to == 'leftHemisphere' and hemi_from == 'rightHemisphere':
            hexels_to[func].left_best_pairings = hexels_from[func].right_best_pairings
        else:
            hexels_to[func].left_best_pairings = hexels_from[func].left_best_pairings


def plot_denoised_vs_noisy(hexels: Dict[str, IPPMHexel], clusterer, title: str):
    """
        Utility function to plot the noisy and denoised versions. It runs the supplied clusterer and then copies the denoised hexels, which
        are fed into a stem plot.

        Parameters
        ----------
        hexels: hexels we want to denoise then plot
        clusterer: A child class of DenoisingStrategy that implements .cluster
        title: title of plot

        Returns
        -------
        Nothing but plots a graph.
    """
    denoised_hexels = clusterer.cluster(hexels, 'rightHemisphere')
    copy_hemisphere(denoised_hexels, hexels, 'leftHemisphere', 'rightHemisphere')
    stem_plot(denoised_hexels, title)
