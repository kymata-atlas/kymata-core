import unittest.mock

from kymata.ippm import denoising_strategies
<<<<<<< HEAD

=======
from kymata.ippm.denoising_strategies import DenoisingStrategy
>>>>>>> 81d3c9a (Refactored Denoiser + Updated tests)

"""
class TestDenoisingStrategies(unittest.TestCase):
    @unittest.mock.patch('denoiser.pd.DataFrame')
    def test_Should_getClusterMins_When_validInputs(self, mock_df):
        # Dataframe has 3 data points with 2 clusters. It should return the second and third rows.
        mock_df.iterrows.return_value = [
            (0, {'Label' : 1, 'Mag' : 1e-10, 'Latency': 100}),
            (1, {'Label' : 2, 'Mag' : 1e-12, 'Latency': 50}),
            (2, {'Label' : 1, 'Mag' : 1e-20, 'Latency': 110})
        ]
        
        denoiser_ = DenoisingStrategy()
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


if __name__ == '__main__':
    unittest.main()
    
"""