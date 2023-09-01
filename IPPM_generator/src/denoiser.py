import numpy as np
from data_tools import Hexel
from typing import Dict, List, Tuple
from copy import deepcopy
from statistics import NormalDist

class MaxPooler(object):
    """
        A denoising strategy based off pooling the maximum spike out of a cluster of spikes.
    """

    def denoise(self, hexels: Dict[str, Hexel], num_clusters=15, bin_sz=25, inplace=False, merge=False) -> Dict[str, Hexel]:
        """
            Takes hexels as input and denoises them using Max Pooling. 

            Algorithm
            --------
                1) Calculate alpha so you can discard all insignificant spikes.
                2) Sort pairings by latency for each function. This is so the spikes are arranged into bins.
                3) For each function
                    3.1) for each pairing in left
                        3.1.1) for each latency_bin in [-200, 800] with step size bin_sz
                            3.1.1.1) If the number of significant spikes > num_clusters then save the maximum spike in the bin else no spike.
                    3.2) Repeat for each pairing in right.

            Params
            ------
                hexels : dictionary containing function names with hexel objects encompassing the pairings
                num_clusters : indicates the number of spikes in a bin for it to become significant. I.e., bins with # of spikes < num_clusters is due to noise.
                bin_sz : pooling length. 
                inplace : overwrite the hexels dict with denoised outputs. 

            Returns
            -------
                Denoised dictionary of Hexels.
        """
        # estimate alpha
        timepoints = 201
        number_of_hexels = 200000
        alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)      # 5-sigma
        bonferroni_corrected_alpha = 1-(pow((1-alpha),(1/(2*timepoints*number_of_hexels))))

        if not inplace:
            # do not overwrite hexels with denoised.
            hexels = deepcopy(hexels)
        
        hexels = self._sort_by_latency(hexels, merge=merge)

        for function in hexels.keys():
            if not merge:
                if len(hexels[function].right_best_pairings) != 0:
                    denoised_pairings = self._pool(hexels[function].right_best_pairings, bonferroni_corrected_alpha, bin_sz, num_clusters)
                    hexels[function].right_best_pairings = denoised_pairings
                
                if len(hexels[function].left_best_pairings) != 0:
                    denoised_pairings = self._pool(hexels[function].left_best_pairings, bonferroni_corrected_alpha, bin_sz, num_clusters)
                    hexels[function].left_best_pairings = denoised_pairings
            
            else:
                # merge hemispheres and return
                hexels[function].merged_best_pairings = self._pool(hexels[function].merged_best_pairings, bonferroni_corrected_alpha, bin_sz, num_clusters)

        return hexels
    
    def _sort_by_latency(self, hexels: Dict[str, Hexel], merge: bool=False):
        """
            Sort pairings by latency in increasing order inplace.

            Params
            ------
                hexels contains all the functions and hexel objects to sort.
            Returns
            -------
                sorted hexels.
        """
        if not merge:
            for function in hexels.keys():
                hexels[function].right_best_pairings.sort(key=lambda x: x[0])
                hexels[function].left_best_pairings.sort(key=lambda x: x[0])
        else:
            for function in hexels.keys():
                hexels[function].merged_best_pairings = hexels[function].right_best_pairings + hexels[function].left_best_pairings
                hexels[function].merged_best_pairings.sort(key=lambda x: x[0])

        return hexels
    
    def _pool(self, pairings: List[Tuple[float, float]], alpha: float, bin_sz: int, num_clusters: int) -> List[Tuple[float, float]]:
        """
            Loop over each bin and check if the number of significant spikes is > num_clusters. If yes,
            save the maximum spike and latency into denoised. Return denoised.

            Params
            ------
                pairings : list of spikes in format (latency, spike mag)
                alpha : significance level
                bin_sz : size of latency bin
                num_clusters : number of significant spikes required to not be classified as noise.
            
            Returns
            -------
                list of denoised pairings.
        """
        idx = 0
        denoised = []
        for latency in range(-200, 800, bin_sz):
            if idx >= len(pairings):
                # reached end of spikes. No point continuing.
                break

            bin_max = np.inf
            lat_max = None
            num_seen = 0 # num of significant spikes seen
            while latency <= pairings[idx][0] < latency + bin_sz:
                mag = pairings[idx][1]
                if mag <= alpha:
                    # significant spike
                    num_seen += 1
                    if mag < bin_max:
                        # the highest peaks are actually the smallest!
                        bin_max = mag
                        lat_max = pairings[idx][0]
                
                idx += 1
                if idx >= len(pairings):
                    # reached end of spikes, break.
                    break
            if bin_max != np.inf and num_seen > num_clusters:
                # if we found more significant spikes than num clusters, we save the maximum.
                denoised.append((lat_max, bin_max))

        return denoised

                

