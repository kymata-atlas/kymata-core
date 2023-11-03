import unittest.mock

import pandas as pd
import numpy as np

from kymata.ippm import denoiser
from kymata.ippm.data_tools import Hexel


class TestDenoisingStrategies(unittest.TestCase):
    # Notes: Need to add test as prefix to run tests.

    def setUp(self):
        self.test_hexels = {
                'func1' : Hexel('func1'),
                'func2' : Hexel('func2'),
                'func3' : Hexel('func3')
        }
        self.test_hexels['func1'].left_best_pairings = [
                (20, 1e-66), (35, 1e-50)
            ]
        self.test_hexels['func2'].left_best_pairings = [
                (10, 1e-2), (25, 1e-56), (75, 1e-80)
            ]
        self.test_hexels['func3'].left_best_pairings = [
                (70, 1e-75), (120, 1e-90), (130, 1e-3)
            ]
        self.test_hexels['func1'].right_best_pairings = [
                (23, 1e-75), (35, 1e-44), (66, 1e-50)
            ]
        self.test_hexels['func2'].right_best_pairings = [
                (45, 1e-60), (75, 1e-55), (80, 1e-45)
            ]
        self.test_hexels['func3'].right_best_pairings = [
                (110, 1e-75), (120, 1e-90)
            ]

    def test_Should_MaxPoolerHaveDefaultParams_When_noParams(self):
        pooler = denoiser.MaxPooler()
        self.assertEqual(pooler._threshold, 15)
        self.assertEqual(pooler._bin_sz, 25)

    def test_Should_maxPoolerHaveCustomParams_When_params(self):
        custom_threshold = 35
        custom_bin_sz = 10
        pooler = denoiser.MaxPooler(threshold=custom_threshold, bin_sz=custom_bin_sz)
        self.assertEqual(pooler._threshold, custom_threshold)
        self.assertEqual(pooler._bin_sz, custom_bin_sz)

    def test_Should_maxPoolerThrowError_When_invalidParams(self):
        self.assertRaises(ValueError, denoiser.MaxPooler, bin_sz='abc')
        self.assertRaises(ValueError, denoiser.MaxPooler, threshold=False)
    
    def test_Should_gmmHaveDefaultParams_When_noParams(self):
        gmm = denoiser.GMM()
        default_vals = [6, 'full', 100, 3, 'k-means++', None]
        actual_vals = [gmm._max_gaussians, gmm._covariance_type, gmm._max_iter, gmm._n_init, gmm._init_params, gmm._random_state]
        self.assertEqual(default_vals, actual_vals)
    
    def test_Should_gmmHaveCustomParams_When_params(self):
        custom_max_gaussians = 4 
        custom_covar_type = 'spherical'
        custom_max_iter =150
        custom_n_init = 5
        custom_init_params = 'kmeans'
        custom_random_state = 1
        gmm = denoiser.GMM(max_gaussians=custom_max_gaussians, covariance_type=custom_covar_type, max_iter=custom_max_iter, n_init=custom_n_init,
                           init_params=custom_init_params, random_state=custom_random_state)
        predicted_vals = [custom_max_gaussians, custom_covar_type, custom_max_iter, custom_n_init, custom_init_params, custom_random_state]
        actual_vals = [gmm._max_gaussians, gmm._covariance_type, gmm._max_iter, gmm._n_init, gmm._init_params, gmm._random_state]
        self.assertEqual(predicted_vals, actual_vals)

    def test_Should_gmmThrowError_When_invalidParams(self):
        self.assertRaises(ValueError, denoiser.GMM, max_gaussians='a')
        self.assertRaises(ValueError, denoiser.GMM, covariance_type='b')
        self.assertRaises(ValueError, denoiser.GMM, max_iter='c')
        self.assertRaises(ValueError, denoiser.GMM, n_init='d')
        self.assertRaises(ValueError, denoiser.GMM, init_params='e')
        self.assertRaises(ValueError, denoiser.GMM, random_state='f')
    
    def test_Should_dbscanHaveDefaultParams_When_noParams(self):
        dbscan = denoiser.DBSCAN()
        default_vals = [10, 2, 'euclidean', None, 'auto', 30, -1]
        params = dbscan._clusterer.get_params(deep=False)
        actual_vals = [params['eps'], params['min_samples'], params['metric'], params['metric_params'], params['algorithm'],
                       params['leaf_size'], params['n_jobs']]
        self.assertEqual(default_vals, actual_vals)

    def test_Should_dbscanHaveCustomParams_When_params(self):
        custom_eps = 15
        custom_min_samples = 3
        custom_metric = 'cosine'
        custom_metric_params = None
        custom_algorithm = 'kd_tree'
        custom_leaf_size = 15
        custom_n_jobs = -1
        dbscan = denoiser.DBSCAN(eps=custom_eps, min_samples=custom_min_samples, metric=custom_metric, metric_params=custom_metric_params,
                                 algorithm=custom_algorithm, leaf_size=custom_leaf_size, n_jobs=custom_n_jobs)
        predicted_vals = [custom_eps, custom_min_samples, custom_metric, custom_metric_params, custom_algorithm,
                          custom_leaf_size, custom_n_jobs]
        params = dbscan._clusterer.get_params(deep=False)
        actual_vals = [params['eps'], params['min_samples'], params['metric'], params['metric_params'], params['algorithm'],
                       params['leaf_size'], params['n_jobs']]
        self.assertEqual(predicted_vals, actual_vals)

    def test_Should_dbscanThrowError_When_invalidParams(self):
        self.assertRaises(ValueError, denoiser.DBSCAN, eps='a')
        self.assertRaises(ValueError, denoiser.DBSCAN, min_samples='b')
        self.assertRaises(ValueError, denoiser.DBSCAN, metric=1)
        self.assertRaises(ValueError, denoiser.DBSCAN, metric_params='d')
        self.assertRaises(ValueError, denoiser.DBSCAN, algorithm='e')
        self.assertRaises(ValueError, denoiser.DBSCAN, leaf_size='f')
        self.assertRaises(ValueError, denoiser.DBSCAN, n_jobs='g')
    
    def test_Should_meanShiftHaveDefaultParams_When_noParams(self):
        mean_shift = denoiser.MeanShift()
        params = mean_shift._clusterer.get_params(deep=False)
        default_vals = [False, None, None, 2, -1]
        actual_vals = [params['cluster_all'], params['bandwidth'], params['seeds'], params['min_bin_freq'], params['n_jobs']]
        self.assertEqual(default_vals, actual_vals)

    def test_Should_meanShiftHaveCustomParams_When_params(self):
        custom_cluster_all = True
        custom_bandwidth = 2
        custom_seeds = None
        custom_min_bin_freq = 3
        custom_n_jobs = -1
        mean_shift = denoiser.MeanShift(cluster_all=custom_cluster_all, bandwidth=custom_bandwidth, seeds=custom_seeds,
                                        min_bin_freq=custom_min_bin_freq, n_jobs=custom_n_jobs)
        params = mean_shift._clusterer.get_params(deep=False)
        predicted_vals = [custom_cluster_all, custom_bandwidth, custom_seeds, custom_min_bin_freq, custom_n_jobs]
        actual_vals = [params['cluster_all'], params['bandwidth'], params['seeds'], params['min_bin_freq'], params['n_jobs']]
        self.assertEqual(predicted_vals, actual_vals)

    def test_Should_meanShiftThrowException_When_invalidParams(self):
        self.assertRaises(ValueError, denoiser.MeanShift, cluster_all='a')
        self.assertRaises(ValueError, denoiser.MeanShift, bandwidth='b')
        self.assertRaises(ValueError, denoiser.MeanShift, seeds=1)
        self.assertRaises(ValueError, denoiser.MeanShift, min_bin_freq='d')
        self.assertRaises(ValueError, denoiser.MeanShift, n_jobs='e')

    def test_Should_checkHemiReturn_When_validParams(self):
        clusterer = denoiser.DenoisingStrategy()
        try:
            clusterer._check_hemi('rightHemisphere')
            clusterer._check_hemi('leftHemisphere')
        except ValueError:
            self.fail('check_hemi failed for valid params.')

    def test_Should_checkHemiThrowException_When_invalidParams(self):
        clusterer = denoiser.DenoisingStrategy()
        self.assertRaises(ValueError, clusterer._check_hemi, 'right hemisphere')

    @unittest.mock.patch('denoiser.pd.DataFrame')
    def test_Should_getClusterMins_When_validInputs(self, mock_df):
        """
            Dataframe has 3 data points with 2 clusters. It should return the second and third rows.
        """
        mock_df.iterrows.return_value = [
            (0, {'Label' : 1, 'Mag' : 1e-10, 'Latency': 100}),
            (1, {'Label' : 2, 'Mag' : 1e-12, 'Latency': 50}),
            (2, {'Label' : 1, 'Mag' : 1e-20, 'Latency': 110})
        ]
        
        denoiser_ = denoiser.DenoisingStrategy()
        actual_vals = denoiser_._get_cluster_mins(mock_df)
        predicted_vals = [(110, 1e-20), (50, 1e-12)]
        self.assertEqual(set(predicted_vals), set(actual_vals))
    
    @unittest.mock.patch('denoiser.pd.DataFrame')
    def test_Should_getClusterMins_When_anomalousInput(self, mock_df):
        mock_df.iterrows.return_value = [
            (0, {'Label' : -1, 'Mag' : 1e-10, 'Latency': 100}),
            (1, {'Label' : 1, 'Mag' : 1e-12, 'Latency': 50}),
        ]

        denoiser_ = denoiser.DenoisingStrategy()
        actual_vals = denoiser_._get_cluster_mins(mock_df)
        predicted_vals = [(50, 1e-12)]
        self.assertEqual(set(predicted_vals), set(actual_vals))


    """
        At this point, I realised that mocking Pandas will get incredibly complex for cases such as iloc. As a result, I believe it is justifiable to assume
        that Pandas will work correctly as it is an extremely widely used library and will have it's own suite of unit tests. From henceforth, I will pass in sample
        dfs instead of mocking.
    """

    def test_Should_clusterBin_When_validInput(self):
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        df.loc[len(df)] = [205, 1e-10]
        df.loc[len(df)] = [210, 1e-15]
        df.loc[len(df)] = [215, 1e-9]
        df.loc[len(df)] = [220, 1e-20]

        denoiser_ = denoiser.MaxPooler(threshold=3) # bin_sz = 25 by default, threshold = 3, so we dont need large df.
        bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=0, latency=200)
        self.assertEqual(bin_min, 1e-20)
        self.assertEqual(lat_min, 220)
        self.assertEqual(num_seen, 4)
        self.assertEqual(r_idx, 4)

    def test_Should_clusterBinTerminate_When_outOfLatencyBin(self):
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        df.loc[len(df)] = [205, 1e-10]
        df.loc[len(df)] = [210, 1e-15]
        df.loc[len(df)] = [215, 1e-9]
        df.loc[len(df)] = [250, 1e-20]

        denoiser_ = denoiser.MaxPooler(threshold=3) # bin_sz = 25, threshold = 15 by default
        bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=0, latency=200)
        self.assertEqual(bin_min, 1e-15)
        self.assertEqual(lat_min, 210) # 1e-20 is out of latency bin, so it should terminate early
        self.assertEqual(num_seen, 3)
        self.assertEqual(r_idx, 3)

    def test_Should_clusterBinsExit_When_emptyBin(self):
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        df.loc[len(df)] = [205, 1e-10]
        df.loc[len(df)] = [210, 1e-15]
        df.loc[len(df)] = [215, 1e-9]
        df.loc[len(df)] = [250, 1e-20]

        denoiser_ = denoiser.MaxPooler(threshold=3) # bin_sz = 25, threshold = 15 by default
        bin_min, lat_min, num_seen, r_idx = denoiser_._cluster_bin(df, r_idx=4, latency=275)
        self.assertEqual(bin_min, float('inf'))
        self.assertEqual(lat_min, None) # 1e-20 is out of latency bin, so it should terminate early
        self.assertEqual(num_seen, 0)
        self.assertEqual(r_idx, 4)

    def test_Should_estimateAlpha(self):
        alpha = denoiser.DenoisingStrategy()._estimate_alpha()
        predicted_alpha = 3.55e-15
        self.assertAlmostEqual(alpha, predicted_alpha)

    def test_Should_filterSpikes_When_validInput(self):
        clusterer = denoiser.DenoisingStrategy()
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        df = clusterer._filter_spikes(self.test_hexels['func2'].left_best_pairings, df, alpha=3e-15)
        self.assertEqual(df.iloc[0, 0], 25)
        self.assertEqual(df.iloc[0, 1], 1e-56)
        self.assertEqual(df.iloc[1, 0], 75)
        self.assertEqual(df.iloc[1, 1], 1e-80)

    def test_Should_filterSpikes_When_emptyInput(self):
        self.test_hexels['func1'].right_best_pairings = []
        clusterer = denoiser.DenoisingStrategy()
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        df = clusterer._filter_spikes(self.test_hexels['func1'].right_best_pairings, df, alpha=3e-15)
        self.assertTrue(len(df) == 0)

    def test_Should_filterSpikes_When_noSignificantSpikes(self):
        clusterer = denoiser.DenoisingStrategy()
        df = pd.DataFrame(columns=['Latency', 'Mag'])
        # increase alpha so all are insignificant.
        df = clusterer._filter_spikes(self.test_hexels['func2'].left_best_pairings, df, alpha=3e-100)
        self.assertTrue(len(df) == 0)

    def test_Should_hexelsToDf_When_validInput(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        test_hexels['f1'].right_best_pairings = [(10, 1e-25), (25, 1e-23)]
        test_hexels['f2'].right_best_pairings = [(50, 1e-10), (55, 1e-44)]
        denoiser_ = denoiser.DenoisingStrategy()
        fs = ['f1', 'f2']
        i = 0
        for func, df in denoiser_._hexels_to_df(test_hexels, 'rightHemisphere'):
            self.assertEqual(fs[i], func)
            if fs[i] == 'f2':
                self.assertEqual(test_hexels['f2'].right_best_pairings[1][0], df.iloc[0, 0])
                self.assertEqual(test_hexels['f2'].right_best_pairings[1][1], df.iloc[0, 1])
            else:
                self.assertEqual(test_hexels['f1'].right_best_pairings[0][0], df.iloc[0, 0])
                self.assertEqual(test_hexels['f1'].right_best_pairings[0][1], df.iloc[0, 1])
                self.assertEqual(test_hexels['f1'].right_best_pairings[1][0], df.iloc[1, 0])
                self.assertEqual(test_hexels['f1'].right_best_pairings[1][1], df.iloc[1, 1])
            i += 1
        
    def test_Should_hexelsToDf_When_leftHemisphere(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        test_hexels['f1'].left_best_pairings = [(10, 1e-25), (25, 1e-23)]
        test_hexels['f2'].left_best_pairings = [(50, 1e-10), (55, 1e-44)]
        denoiser_ = denoiser.DenoisingStrategy()
        fs = ['f1', 'f2']
        i = 0
        for func, df in denoiser_._hexels_to_df(test_hexels, 'leftHemisphere'):
            self.assertEqual(fs[i], func)
            if fs[i] == 'f2':
                self.assertEqual(test_hexels['f2'].left_best_pairings[1][0], df.iloc[0, 0])
                self.assertEqual(test_hexels['f2'].left_best_pairings[1][1], df.iloc[0, 1])
            else:
                self.assertEqual(test_hexels['f1'].left_best_pairings[0][0], df.iloc[0, 0])
                self.assertEqual(test_hexels['f1'].left_best_pairings[0][1], df.iloc[0, 1])
                self.assertEqual(test_hexels['f1'].left_best_pairings[1][0], df.iloc[1, 0])
                self.assertEqual(test_hexels['f1'].left_best_pairings[1][1], df.iloc[1, 1])
            i += 1

    def test_Should_hexelsToDf_When_emptyInput(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        denoiser_ = denoiser.DenoisingStrategy()
        for _, df in denoiser_._hexels_to_df(test_hexels, 'leftHemisphere'):
            self.assertTrue(len(df) == 0)

    def test_Should_updatePairings_When_validInput(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        test_hexels['f1'].right_best_pairings = [(10, 1e-20), (15, 1e-2)]
        test_hexels['f2'].right_best_pairings = [(200, 1e-1), (250, 1e-50)]
        denoised = [(10, 1e-20)]
        denoiser_ = denoiser.DenoisingStrategy()
        hexels = denoiser_._update_pairings(test_hexels, 'f1', denoised, 'rightHemisphere')
        self.assertEqual(hexels['f1'].right_best_pairings, denoised)
        self.assertEqual(hexels['f2'].right_best_pairings, [(200, 1e-1), (250, 1e-50)])

    def test_Should_updatePairings_When_leftHemisphere(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        test_hexels['f1'].left_best_pairings = [(10, 1e-20), (15, 1e-2)]
        test_hexels['f2'].left_best_pairings = [(200, 1e-1), (250, 1e-50)]
        denoised = [(10, 1e-20)]
        denoiser_ = denoiser.DenoisingStrategy()
        hexels = denoiser_._update_pairings(test_hexels, 'f1', denoised, 'leftHemisphere')
        self.assertEqual(hexels['f1'].left_best_pairings, denoised)
        self.assertEqual(hexels['f2'].left_best_pairings, [(200, 1e-1), (250, 1e-50)])
    
    def test_Should_maxPoolerCluster_When_validInput(self):
        pooler = denoiser.MaxPooler(bin_sz=50, threshold=1)
        denoised = pooler.cluster(self.test_hexels, 'rightHemisphere')
        self.assertEqual(denoised['func1'].right_best_pairings, [(23, 1e-75), (66, 1e-50)])
        self.assertEqual(denoised['func2'].right_best_pairings, [(45, 1e-60), (75, 1e-55)])
        self.assertEqual(denoised['func3'].right_best_pairings, [(120, 1e-90)])

    def test_Should_maxPoolerCluster_When_noSignificantBins(self):
        # all bins < threshold for significance.
        pooler = denoiser.MaxPooler(bin_sz=50)
        denoised = pooler.cluster(self.test_hexels, 'rightHemisphere')
        self.assertEqual(denoised['func1'].right_best_pairings, [])
        self.assertEqual(denoised['func2'].right_best_pairings, [])
        self.assertEqual(denoised['func3'].right_best_pairings, [])

    def test_Should_maxPoolerCluster_When_emptyInput(self):
        test_hexels = {'f1' : Hexel('f1'), 'f2' : Hexel('f2')}
        test_hexels['f1'].right_best_pairings = []
        test_hexels['f2'].right_best_pairings = []

        pooler = denoiser.MaxPooler()
        denoised = pooler.cluster(test_hexels, 'rightHemisphere')
        self.assertEqual(denoised['f1'].right_best_pairings, [])
        self.assertEqual(denoised['f2'].right_best_pairings, [])

    def test_Should_gmmCluster_When_validInputRightHemisphere(self):
        """
            What we expect to happen: Each data point will be it's own cluster, except points that are identified as insignificant.
            Why: The likelihood is maximised by assigning each point to it's own cluster. It is the side-effect
                 of having too little data points. 
        """
        np.random.seed(0) # for reproducibility of ML algorithms
        clusterer = denoiser.GMM()
        denoised = clusterer.cluster(self.test_hexels, 'rightHemisphere')
        f2_expected = self.test_hexels['func2'].right_best_pairings
        self.assertEqual(denoised['func1'].right_best_pairings, self.test_hexels['func1'].right_best_pairings)
        self.assertEqual(denoised['func2'].right_best_pairings, f2_expected)
        self.assertEqual(denoised['func3'].right_best_pairings, self.test_hexels['func3'].right_best_pairings)

    def test_Should_gmmCluster_When_validInputLeftHemisphere(self):
        """
            What we expect to happen: Each data point will be it's own cluster, except points that are identified as insignificant.
            Why: The likelihood is maximised by assigning each point to it's own cluster. It is the side-effect
                 of having too little data points. 

            E.g., (10, 0.01) is identified as insignificant since it is greater than alpha. Hence, exclude.
        """
        np.random.seed(0) # for reproducibility of ML algorithms
        clusterer = denoiser.GMM()
        denoised = clusterer.cluster(self.test_hexels, 'leftHemisphere')
        f2_expected = self.test_hexels['func2'].left_best_pairings
        f3_expected = self.test_hexels['func3'].left_best_pairings
        f2_expected.remove((10, 0.01))
        f3_expected.remove((130, 0.001))
        self.assertEqual(denoised['func1'].left_best_pairings, self.test_hexels['func1'].left_best_pairings)
        self.assertEqual(denoised['func2'].left_best_pairings, self.test_hexels['func2'].left_best_pairings)
        self.assertEqual(denoised['func3'].left_best_pairings, self.test_hexels['func3'].left_best_pairings)

    def test_Should_gmmCluster_When_singleDataPoint(self):
        np.random.seed(0)
        test_hexels = {'f1' : Hexel('f1')}
        test_hexels['f1'].right_best_pairings = [(10, 1e-20)]
        clusterer = denoiser.GMM()
        denoised = clusterer.cluster(test_hexels, 'rightHemisphere')
        self.assertEqual(denoised['f1'].right_best_pairings, [(10, 1e-20)])
        
    def test_Should_dbscanCluster_When_validInputRightHemi(self):
        np.random.seed(0)
        clusterer = denoiser.DBSCAN()
        denoised = clusterer.cluster(self.test_hexels, 'rightHemisphere')
        self.assertEqual([], denoised['func1'].right_best_pairings)
        self.assertEqual([(75, 1e-55)], denoised['func2'].right_best_pairings)
        self.assertEqual([(120, 1e-90)], denoised['func3'].right_best_pairings)

    def test_Should_dbscanCluster_When_validInputLeftHemi(self):
        """
            The reason DBSCAN returns empty is that a point is only considered significant if it has at least
            one more point within a 10 ms radius. In this case, none do. Moreover, the magnitudes that are higher than
            alpha are excluded as insignificant.
        """
        np.random.seed(0)
        clusterer = denoiser.DBSCAN()
        denoised = clusterer.cluster(self.test_hexels, 'leftHemisphere')
        self.assertEqual([], denoised['func1'].left_best_pairings)
        self.assertEqual([], denoised['func2'].left_best_pairings)
        self.assertEqual([], denoised['func3'].left_best_pairings)

    def test_Should_meanShiftCluster_When_validInputRightHemi(self):
        np.random.seed(0)
        clusterer = denoiser.MeanShift()
        denoised = clusterer.cluster(self.test_hexels, 'rightHemisphere')
        self.assertEqual(self.test_hexels['func1'].right_best_pairings, denoised['func1'].right_best_pairings)
        self.assertEqual(self.test_hexels['func2'].right_best_pairings, denoised['func2'].right_best_pairings)
        self.assertEqual(self.test_hexels['func3'].right_best_pairings, denoised['func3'].right_best_pairings)

    def test_Should_meanShiftCluster_When_validInputLeftHemi(self):
        np.random.seed(0)
        clusterer = denoiser.MeanShift()
        denoised = clusterer.cluster(self.test_hexels, 'leftHemisphere')
        f2_expected = self.test_hexels['func2'].left_best_pairings
        f3_expected = self.test_hexels['func3'].left_best_pairings
        f2_expected.remove((10, 0.01))
        f3_expected.remove((130, 0.001))
        self.assertEqual(self.test_hexels['func1'].left_best_pairings, denoised['func1'].left_best_pairings)
        self.assertEqual(f2_expected, denoised['func2'].left_best_pairings)
        self.assertEqual(self.test_hexels['func3'].left_best_pairings, denoised['func3'].left_best_pairings)

if __name__ == '__main__':
    unittest.main()