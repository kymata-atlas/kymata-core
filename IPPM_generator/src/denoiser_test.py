import unittest
import unittest.mock 
import denoiser
from data_tools import Hexel

class TestDenoiser(unittest.TestCase):

    def test_max_pooler_sort(self):
        test_hexels = {'func1' : Hexel('func1')}
        test_hexels['func1'].left_best_pairings = [(153, 21), (12, 47), (-78, 435)]
        test_hexels['func1'].right_best_pairings = [(656, 35), (-200, 34), (51, 66)]
        # should re-order so latency is increasing
        pooler = denoiser.MaxPooler()
        sorted_hex = pooler._sort_by_latency(test_hexels)
        
        self.assertEqual(sorted_hex['func1'].left_best_pairings, [(-78, 435), (12, 47), (153, 21)])
        self.assertEqual(sorted_hex['func1'].right_best_pairings, [(-200, 34), (51, 66), (656, 35)])

    def test_max_pooler_pool(self):
        pairing = [(10, 55), (35, 123), (45, 152), (56, 33), (76, 36), (110, 23), (125, 55)]
        pooler = denoiser.MaxPooler() # in the graphs, the highest peaks are actually the smallest
        denoised = pooler._pool(pairing, 500, 50)

        self.assertEqual(denoised, [(10, 55), (56, 33), (110, 23)])

    @unittest.mock.patch('denoiser.np')
    @unittest.mock.patch('denoiser.NormalDist')
    def test_max_pooler(self, mock_dist, mock_np):
        """ create artifical data. Should see the following:
            left hemi: [0, 50) Most significant is f1 : 35, f2: 10, f3: N/A
                     [50, 100) Most significant is f1: N/A, f2: 75, f3: 70
                     [100, 150) Most significant is f1: N/A, f2: N/A, f3: 130
            right hemi: [0, 50) Most significant is f1: 35, f2: 45, f3: N/A
                    [50, 100) Most significant is f1: 66, f2: 80, f3: N/A
                    [100, 150) Most significant is f1: N/A, f2: N/A, f3: 110.
        
            other notes
            -----------
            Need to mock statistics.NormalDist. It returns 0.9999997133484282
            Mock numpy.inf
         """
        mock_dist.return_value.cdf.return_value = 0.9999997133484282
        mock_np.inf = 100000 # arbitrary large value greater than all mags
        test_hexels = {
                'func1' : Hexel('func1'),
                'func2' : Hexel('func2'),
                'func3' : Hexel('func3')
        }
        test_hexels['func1'].left_best_pairings = [
                (20, 1e-66), (35, 1e-50)
            ]
        test_hexels['func2'].left_best_pairings = [
                (10, 1e-2), (25, 1e-56), (75, 1e-80)
            ]
        test_hexels['func3'].left_best_pairings = [
                (70, 1e-75), (120, 1e-90), (130, 1e-3)
            ]
        test_hexels['func1'].right_best_pairings = [
                (23, 1e-75), (35, 1e-44), (66, 1e-50)
            ]
        test_hexels['func2'].right_best_pairings = [
                (45, 1e-60), (75, 1e-55), (80, 1e-45)
            ]
        test_hexels['func3'].right_best_pairings = [
                (110, 1e-75), (120, 1e-90)
            ]
        
        pooler = denoiser.MaxPooler()
        denoised = pooler.denoise(test_hexels, bin_sz=50, inplace=True)
        self.assertEqual(denoised['func1'].left_best_pairings, [(20, 1e-66)])
        self.assertEqual(denoised['func2'].left_best_pairings, [(25, 1e-56), (75, 1e-80)])
        self.assertEqual(denoised['func3'].left_best_pairings, [(70, 1e-75), (120, 1e-90)])
        self.assertEqual(denoised['func1'].right_best_pairings, [(23, 1e-75), (66, 1e-50)])
        self.assertEqual(denoised['func2'].right_best_pairings, [(45, 1e-60), (75, 1e-55)])
        self.assertEqual(denoised['func3'].right_best_pairings, [(120, 1e-90)])
        
if __name__ == '__main__':
    unittest.main()