import pandas as pd
import numpy as np

from sklearn.cluster import KMeans as KMeans_, DBSCAN as DBSCAN_, MeanShift as MeanShift_
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import data_tools
from data_tools import Hexel
from statistics import NormalDist
from typing import Dict, List, Tuple
from copy import deepcopy

class DenoisingStrategy(object):
    """
        Interface for unsupervised clustering algorithms. Strategies should conform to this interface.
    """
    def __init__(self, **kwargs):
        """
            kwargs varies from strategy to strategy. The subclasses should be aware of the important ones and set default values if
            there is no value supplied for it. Although sklearn has guardrails for input params, some assertions are also provided.
        """
        self._clusterer = None
    
    def cluster(self, hexels: Dict[str, Hexel], hemi: str) -> Dict[str, Hexel]:
        """
            For each function in hemi, it will attempt to construct a dataframe that holds significant spikes (i.e., abova alpha).
            Next, it clusters using self._clusterer. Finally, it locates the minimum (most significant) point for each cluster and saves
            it. 

            This can be overridden if using a custom clustering strategy but as it is, it works well sklearn clustering techniques. As a result,
            additional algorithms from sklearn can be easily incorporated.

            Params
            ------
            hexels : Dict[str, Hexel]
                     Contains the left hemisphere and right hemisphere pairings. We want to denoise one of them.
            hemi : str from ['rightHemisphere', 'leftHemisphere']
                   indicates the hemisphere to denoise.

            Returns
            -------
            A new dictionary that has the denoised pairings for each function. Same format as input dict hexels.
            
        """
        self._check_hemi(hemi) # guardrail to check for hemisphere input.
        hexels = deepcopy(hexels) # dont do it in-place.
        for func, df in self._hexels_to_df(hexels, hemi):
            if len(df) == 0:
                # there are no significant spikes.
                hexels = self._update_pairings(hexels, func, [], hemi)
                continue

            fitted = self._clusterer.fit(df)
            df['Label'] = fitted.labels_
            cluster_mins = self._get_cluster_mins(df)
            hexels = self._update_pairings(hexels, func, cluster_mins, hemi)
        return hexels
    
    def _hexels_to_df(self, hexels: Dict[str, Hexel], hemi: str) -> pd.DataFrame:
        """
            A generator used to build a dataframe of significant points only. For each call, it returns the dataframe
            for the next function in hexels.keys().

            Params
            ------
            hexels : dict[str, Hexel]
                     A dictionary with function names as keys and hexel objects containing the pairings. 
            hemi : str from ['rightHemisphere', 'leftHemisphere']

            Returns
            -------
            A dataframe for a function that contains only significant spikes, i.e., those above alpha.
        """
        alpha = self._estimate_alpha()
        for func in hexels.keys():
            df = pd.DataFrame(columns=['Latency', 'Mag'])
            df = self._filter_spikes(hexels[func].right_best_pairings if hemi == 'rightHemisphere' else
                                     hexels[func].left_best_pairings,
                                     df, 
                                     alpha)
            yield (func, df)

    def _filter_spikes(self, spikes: List[Tuple[float, float]], df: pd.DataFrame, alpha: float) -> pd.DataFrame:
        """
            Use this to filter out all insignificant spikes and add significant spikes to a dataframe.

            Params
            ------
            spikes : List[float, float]
                     the pairings from either the left or right hemisphere. Collection of (latency, magnitude) pairs.
            df : pd.DataFrame
                 A dataframe to which we want to add the significant spikes. It has columns [Latency, Mag].
            alpha : float
                    The threshold for significance.

            Returns
            -------
            the df parameter but populated with significant spikes.
        """
        for latency, spike in spikes:
            if spike <= alpha:
                #significant
                df.loc[len(df)] = [latency, spike]
        return df

    def _estimate_alpha(self) -> float:
        """
            Fetch the threshold of significance. Currently hard-coded but can be updated to take params. IF custom one is required,
            it might be better to move estimate alpha to data_utils and precompute alpha and pass as a variable to clustering algos.

            Returns
            -------
            A float indicating the magnitude that a spike needs to be smaller than for it to be significant.
        """
        timepoints = 201
        number_of_hexels = 200000
        alpha = 1 - NormalDist(mu=0, sigma=1).cdf(5)      # 5-sigma
        bonferroni_corrected_alpha = 1-(pow((1-alpha),(1/(2*timepoints*number_of_hexels))))
        return bonferroni_corrected_alpha
    
    def _check_hemi(self, hemi):
        """
            A guardrail function to verify the hemi input is one of rightHemisphere or leftHemisphere.

            Params
            ------
            hemi : str
                   should be rightHemisphere or leftHemisphere
            
            Throws
            ------
            ValueError if invalid hemi input.
        """
        if not hemi in ['rightHemisphere', 'leftHemisphere']:
            print('Hemisphere needs to be rightHemisphere or leftHemisphere')
            raise ValueError
    
    def _update_pairings(self, hexels: Dict[str, Hexel], func: str, denoised: List[Tuple[float, float]], hemi: str) -> Dict[str, Hexel]:
        """
            Overwrite the previous pairings with the new denoised version. 

            Params
            ------
            hexels : Dict[func_name, Hexel]
                     Where we want to save the new denoised pairings.
            func : a key for hexels
                   Indicates the function we are saving for. Use it to index into hexels.
            denoised : List[(latency, magnitude)]
                       The denoised pairings. 
            hemi : rightHemisphere or leftHemisphere
                   the hemisphere we are denoising over.

            Returns
            -------
            denoised hexels.
        """
        if hemi == 'rightHemisphere':
            hexels[func].right_best_pairings = denoised
        else:
            hexels[func].left_best_pairings = denoised
        return hexels

    def _get_cluster_mins(self, df: pd.DataFrame) -> List[Tuple[float, float]]:
        """
            Use this to construct a list of tuples that contain the most significant (minimum) spikes for each class.
            Note: assume that a label of -1 is an anomaly. Some of the algorithms in sklearn can identify outliers, so it is a useful assumption.

            Params
            ------
            df : pd.DataFrame
                 It needs to have a column called labels, which contains the cluster assignment for each data point.
            
            Returns
            -------
            For each class, the most significant spike and the associated latency.
        """
        ret = []
        class_mins = {}
        for _, row in df.iterrows():
            label = row['Label']
            if label == -1:
                # label = -1 indicates anomaly, so exclude.
                continue
                
            if not label in class_mins.keys():
                # first time seeing this class.
                class_mins[label] = [float(row['Mag']), float(row['Latency'])]
            elif class_mins[label][0] > row['Mag']:
                # found a more significant spike, so overwrite.
                class_mins[label] = [float(row['Mag']), float(row['Latency'])]
            
        for _, items in class_mins.items():
            # (latency, magnitude)
            ret.append((items[1], items[0]))

        return ret
    
class MaxPooler(DenoisingStrategy):
    """
        Naive max pooling technique. It operates by first sorting the latencies into bins, identifying significant bins, and taking the most significant spikes in a significant bin.
        A bin is considered significant if the number of spikes for a particular function in the bin exceeds the threshold (self._threshold). Moreover the bin size can be controlled by the
        bin_sz hyperparameter. Hence, to improve robustness, the threshold should be increased or bin size reduced. A criteria that is too stringent may lead to no significant spikes, so it should be balanced.
        Finally, it is possible to run max pooler as an anomaly detection system prior to running an unsupervised algorithm, albeit at a higher computational cost.
    """
    
    def __init__(self, **kwargs):
        """
            Params
            ------
            threshold : int
                        # of spikes required in a bin before it is considered significant
            bin_sz : int
                     the size, in ms, of a bin.
        """
        
        self._threshold = 15 if not 'threshold' in kwargs.keys() else kwargs['threshold']
        self._bin_sz = 25 if not 'bin_sz' in kwargs.keys() else kwargs['bin_sz']
        if type(self._threshold) != int:
            print('Threshold needs to be an integer.')
            raise ValueError
        if type(self._bin_sz) != int:
            print('Bin size needs to be an integer.')
            raise ValueError

    def cluster(self, hexels: Dict[str, Hexel], hemi: str) -> List[Tuple[float, float]]:
        """  
            Custom clustering method since it differs from other unsupervised techniques. 

            Algorithm
            ---------
                sort by latency, so we can partition into bins in ascending order. If it is unordered, then there is no guarentee that adjacent data points belong to the same bin.
                ret = []
                for current_bin in partitioned_latency_axis:
                    number_of_spikes := 0
                    most_significant := infinity (the most significant are the ones with the smallest magnitude)
                    latency := null
                    for (current_data_point, current_latency) in current_bin
                        number_of_spikes++
                        if current_data_point < most_significant:
                            most_significant = current_data_point
                            latency = current_latency
                        
                    if number_of_spikes > threshold:
                        ret.append((latency, most_significant))
                return ret                    

            Params
            ------
            see DenoisingStrategy.    

            Returns
            -------
            see DenoisingStrategy.
        """
        super()._check_hemi(hemi)
        hexels = deepcopy(hexels)
        for func, df in super()._hexels_to_df(hexels, hemi):
            if len(df) == 0:
                hexels = super()._update_pairings(hexels, func, [], hemi)
                continue

            df = df.sort_values(by='Latency') # arrange latencies into bins
            r_idx = 0                         # this points to the current_data_point. It is incremented in the inner loop.
            ret = []
            for latency in range(-200, 800, self._bin_sz):
                # latency indicates the start of the current bin. E.g., latency = -200 means the bin is [-200, -200 + self._bin_sz)
                if r_idx >= len(df):
                    # no data left.
                    break

                bin_min, lat_min, num_seen, r_idx = self._cluster_bin(df, r_idx, latency)

                if bin_min != np.inf and num_seen >= self._threshold:
                    # significant bin, so save the cluster mins.
                    ret.append((lat_min, bin_min))
            
            hexels = super()._update_pairings(hexels, func, ret, hemi)

        return hexels
    
    def _cluster_bin(self, df: pd.DataFrame, r_idx: int, latency: int) -> Tuple[float, int, int, int]:
        """
            We dont need to check r_idx and latency since cluster function provides them rather than the user.

            Params
            ------
            df : pd.DataFrame
                 Holds the data about a cluster, specifically, the latency and magnitude.
            r_idx : int
                    Index into the dataframe. Initially, it points to the start of the bin. Since df is ordered by latency, we just need to increment to get next.
            latency : int
                      latency is the current start of bin latency. E.g., latency = 100 means that we are looping over the bin [100, 100 + bin_sz) and r_idx points to the first element in the bin.

            Returns
            --------
            A tuple with format: (minimum magnitude of this bin, the associated latency with the magnitude, number of significant spikes within this bin, the r_idx pointing to the next bin).
        """

        bin_min = float('inf')
        lat_min = None
        num_seen = 0

        if r_idx >= len(df):
            return bin_min, lat_min, num_seen, r_idx

        while latency <= df.iloc[r_idx, 0] < latency + self._bin_sz:
             # loop while the r_idx lands in the current_bin. Once r_idx points to a data point with latency outside of the bin, we break the loop.
            mag = df.iloc[r_idx, 1]
            num_seen += 1
            if mag < bin_min:
                bin_min = mag
                lat_min = df.iloc[r_idx, 0]

            r_idx += 1
            if r_idx >= len(df):
                break
        
        return bin_min, lat_min, num_seen, r_idx

            
class GMM(DenoisingStrategy):
    """
        This strategy uses the GaussianMixtureModel algorithm. Intuitively, it attempts to fit a multimodal Gaussian distribution to the data using the EM algorithm.
        The primary disadvantage of this model is that the number of Gaussians have to be prespecified. This implementation does a grid search from 1 to max_gaussians 
        to find the optimal number of Gaussians. Moreover, it does not work well with anomalies.
    """
    def __init__(self, **kwargs):
        """
            Params
            ------
            For more details, check https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
            max_gaussians : int
                            the upper bound on the number of Gaussians to search. 
            covariance_type : str
                              the constraints on the covariance matrix. 'full' means a full covariance matrix.
            max_iter : int
                       the maximum number of EM steps before optimisation terminates.
            n_init : int
                     the number of initialisations (reruns) to do for each max_gaussian. GMM is highly dependent on the initialisations, so it makes sense to do multiple and take the best.
            init_params : str
                          the algorithm used to initialise the parameters. 'k-means++' is an improved version of k-means.
            random_state : int
                           set this if you want your results to be reproducible.
        """
        # we are instantiating multiple models, so save hyperparameters instead of clusterer object.
        self._max_gaussians = 6 if not 'max_gaussians' in kwargs.keys() else kwargs['max_gaussians']
        self._covariance_type = 'full' if not 'covariance_type' in kwargs.keys() else kwargs['covariance_type']
        self._max_iter = 100 if not 'max_iter' in kwargs.keys() else kwargs['max_iter']
        self._n_init = 3 if not 'n_init' in kwargs.keys() else kwargs['n_init']
        self._init_params = 'k-means++' if not 'init_params' in kwargs.keys() else kwargs['init_params']
        self._random_state = None if not 'random_state' in kwargs.keys() else kwargs['random_state']

        invalid = False
        if type(self._max_gaussians) != int:
            print('Max Gaussians must be of type int.')
            invalid = True
        if not self._covariance_type in ['full', 'tied', 'diag', 'spherical']:
            print('Covariance type must be one of {full, tied, diag, spherical}')
            invalid = True
        if type(self._max_iter) != int:
            print('Max iterations must be of type int.')
            invalid = True
        if type(self._n_init) != int:
            print('Number of initialisations must be of type int.')
            invalid = True
        if not self._init_params in ['kmeans', 'k-means++', 'random', 'random_from_data']:
            print('Initalisation of parameter strategy must be one of {kmeans, k-means++, random, random_from_data}')
            invalid = True
        if self._random_state != None and type(self._random_state) != int:
            print('Random state must be none or int.')
            invalid = True
        
        if invalid:
            raise ValueError
        

    def cluster(self, hexels: Dict[str, Hexel], hemi: str) -> Dict[str, Hexel]:
        """
            Overriding the superclass cluster function because we want to perform a grid-search over the number of clusters to locate the optimal one.
            It works similarly to the superclass.cluster method but it performs it multiple times. It stops if the number of data points < number of clusters as
            it will not work. 

            Params
            ------
            Hexels : Dict[function_name, hexel_object]
                     The data we want to cluster over.
            hemi : str
                   Needs to be leftHemisphere or rightHemisphere.

            Returns
            -------
            Hexels that are formatted the same as input hexels but with denoised pairings.
        """
        super()._check_hemi(hemi)
        hexels = deepcopy(hexels)
        for func, df in super()._hexels_to_df(hexels, hemi):
            if len(df) == 1:
                # no point clustering, just return the single data point.
                ret = []
                for _, row in df.iterrows():
                    ret.append((row['Latency'], row['Mag']))
                hexels = super()._update_pairings(hexels, func, ret, hemi)
                continue

            best_labels = None
            best_score = np.inf # use aic, bic or silhouette score for model selection. if silhouette, switch this to -inf since we wanna maximise it.
            for n in range(1, self._max_gaussians):
                if n > len(df):
                    # the number of gaussians has to be less than the number of datapoints.
                    continue
                gmm = GaussianMixture(n_components=n, 
                                      covariance_type=self._covariance_type,
                                      max_iter=self._max_iter,
                                      n_init=self._n_init,
                                      init_params=self._init_params,
                                      random_state=self._random_state)
                gmm.fit(df)
                score = gmm.bic(df)  # gmm.aic(df) for AIC score. 
                if score < best_score:
                    # this condition depends on the choice of AIC/BIC/silhouette. if using silhouette, reverse the inequality.
                    best_labels = gmm.predict(df)
                    best_score = score
        
            df['Label'] = best_labels
            cluster_mins = super()._get_cluster_mins(df)
            hexels = super()._update_pairings(hexels, func, cluster_mins, hemi)
        return hexels
    
class DBSCAN(DenoisingStrategy):
    """
        This approach leverages the algorithm known as Density-based clustering, which focuses on the density of points rather than solely the distance. Intuitively, for each data point, it
        constructs an epsilon-ball neighbourhood and attempts to jump to nearby points and so on. Points that are reached through successive jumps are assigned to the same cluster. To make results more robust,
        it only uses data points where there are at least 2 samples in the neighbourhood, enabling DBSCAN to detect outliers. To make the algorithm tractable, a KD-tree is constructed for efficient look-up of nearby
        points.

        Params
        ------
        For more details, https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        eps : int
              the radius of the neighbourhood ball. 
        min_samples : int
                      the number of points required in a neighbourhood for it to be considered significant.
        metric : str
                 the metric used to calculate the distance between points
        metric_params : dict
                        the parameters input to the metric. For instance, if you use Gaussian, you will have to supply the bandwidth parameter.
        algorithm : str
                    the algorithm used to find nearby points. 'auto' chooses the best one itself.
        leaf_size : int
                    leaf size of the KD-tree used for look-up of nearby points. This primarily influences the latency. 
        n_jobs : int
                 the number of processors to use. -1 means use all available ones
    """
    def __init__(self, **kwargs):
        eps = 10 if not 'eps' in kwargs.keys() else kwargs['eps']
        min_samples = 2 if not 'min_samples' in kwargs.keys() else kwargs['min_samples']
        metric = 'euclidean' if not 'metric' in kwargs.keys() else kwargs['metric']
        metric_params = None if not 'metric_params' in kwargs.keys() else kwargs['metric_params']
        algorithm = 'auto' if not 'algorithm' in kwargs.keys() else kwargs['algorithm']
        leaf_size = 30 if not 'leaf_size' in kwargs.keys() else kwargs['leaf_size']
        n_jobs = -1 if not 'n_jobs' in kwargs.keys() else kwargs['n_jobs']

        invalid = False
        if type(eps) != int:
            print('Epsilon must be of type integer.')
            invalid = True
        if type(min_samples) != int:
            print('Min samples must be of type integer.')
            invalid = True
        if type(metric) != str:
            print('Metric must be a string. It should be one from https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html#sklearn.metrics.pairwise_distances')
            invalid = True
        if metric_params != None and type(metric_params) != dict:
            print('Metric params must be a dict or None.')
            invalid = True
        if not algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']:
            print('Algorithm must be one of {auto, ball_tree, kd_tree, brute}')
            invalid = True
        if type(leaf_size) != int:
            print('leaf_size must be of type int.')
            invalid = True
        if type(n_jobs) != int:
            print('The number of jobs must be of type int.')
            invalid = True

        if invalid:
            raise ValueError

        self._clusterer = DBSCAN_(eps=eps, 
                                  min_samples=min_samples,
                                  metric=metric,
                                  metric_params=metric_params,
                                  algorithm=algorithm,
                                  leaf_size=leaf_size,
                                  n_jobs=n_jobs)
    
class MeanShift(DenoisingStrategy):
    """
        The mean shift algorithm is an improved variant of k-means that does not require the number of clusters to be
        prespecified in advance. 

        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html for more details.

        Params
        ------
        cluster_all : bool, default = False
                      whether to exclude anomalies or to cluster all points
        bandwidth : float, default = None
                    the bandwidth of the flat kernel used.
        seeds : list, default = None
                the seeds used to initialise the kernel. If none, it estimates it.
        min_bin_freq : int, default = 2
                       the number of points in a bin before it can be considered significant
        n_jobs : int, default = -1
                 the number of processors used. -1 means use all available processors.
    """
    def __init__(self, **kwargs):
        cluster_all = False if not 'cluster_all' in kwargs.keys() else kwargs['cluster_all']
        bandwidth = None if not 'bandwidth' in kwargs.keys() else kwargs['bandwidth']
        seeds = None if not 'seeds' in kwargs.keys() else kwargs['seeds']
        min_bin_freq = 2 if not 'min_bin_freq' in kwargs.keys() else kwargs['min_bin_freq']
        n_jobs = -1 if not 'n_jobs' in kwargs.keys() else kwargs['n_jobs']
        
        invalid = False
        if type(cluster_all) != bool:
            print('Cluster_all must be of type bool.')
            invalid = True
        if type(bandwidth) != float and type(bandwidth) != int and bandwidth != None:
            print('bandwidth must be None or float.')
            invalid = True
        if type(seeds) != list and seeds != None:
            print('Seeds must be a list or None.')
            invalid = True
        if type(min_bin_freq) != int:
            print('Mininum bin frequency must be of type int.')
            invalid = True
        if type(n_jobs) != int:
            print('Number of jobs must be of type int.')
            invalid = True

        if invalid:
            raise ValueError

        self._clusterer = MeanShift_(bandwidth=bandwidth, 
                                     seeds=seeds,
                                     min_bin_freq=min_bin_freq,
                                     cluster_all=cluster_all,
                                     n_jobs=n_jobs)


