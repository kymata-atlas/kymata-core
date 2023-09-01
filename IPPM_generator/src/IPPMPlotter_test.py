import unittest
import unittest.mock
from data_tools import Hexel
import numpy as np
from IPPMPlotter import IPPMPlotter

class TestIPPMPlotter(unittest.TestCase):

    @unittest.mock.patch('IPPMPlotter.np')
    def test_make_bspline_ctr_points(self, mock_np):
        def my_side_effect(list):
            # use this to mock np.array but return the list instead of np.array
            return list
        # now if np.array is called, it will return the same list.
        mock_np.array.side_effect = my_side_effect
        start_n_end = [(0, 100), (100, 0)] # top left to bottom right
        plotter = IPPMPlotter()
        ctr_points = plotter._make_bspline_ctr_points(start_n_end)
        expected_ctr_points = [(0, 100), (20, 100), (25, 100), (30, 0), (35, 0), (100, 0)]
        self.assertEqual(expected_ctr_points, ctr_points)

    @unittest.mock.patch('IPPMPlotter.np')
    def test_make_bspline_path(self, mock_np):
        def append_side_effect(list1, list2):
            return list1 + list2
        def linspace_side_effect(start, end, step, endpoint):
            return list(np.linspace(start, end, step, endpoint=endpoint))
        def splev_side_effect(u3, tck):
            # not sure how to mock this tbh since the entire function depends on it.
            # could hard-code the response but then the test is trivial since we hard-code the
            # response then verify it.
            return np.splev(u3, tck)
        
        mock_np.linspace.side_effect = linspace_side_effect
        mock_np.append.side_effect = append_side_effect
        mock_np.splev.side_effect = splev_side_effect

        ctr_points = [(0, 100), (20, 100), (25, 100), (30, 0), (35, 0), (100, 0)]
        plotter = IPPMPlotter()
        bspline_path = plotter._make_bspline_path(np.array(ctr_points))
        expected_path = [np.array([  0.        ,   2.51085587,   4.83164845,   6.97080217,
                                    8.93674146,  10.73789074,  12.38267445,  13.879517  ,
                                    15.23684283,  16.46307635,  17.56664201,  18.55596422,
                                    19.43946741,  20.22557601,  20.92271445,  21.53930714,
                                    22.08377853,  22.56455303,  22.99005507,  23.36870908,
                                    23.70893948,  24.01917071,  24.30782718,  24.58333333,
                                    24.85277801,  25.11790773,  25.37913345,  25.63686611,
                                    25.89151667,  26.14349607,  26.39321525,  26.64108518,
                                    26.88751678,  27.13292102,  27.37770883,  27.62229117,
                                    27.86707898,  28.11248322,  28.35891482,  28.60678475,
                                    28.85650393,  29.10848333,  29.36313389,  29.62086655,
                                    29.88209227,  30.14722199,  30.41666667,  30.69587135,
                                    31.01041752,  31.3909208 ,  31.86799677,  32.47226103,
                                    33.2343292 ,  34.18481685,  35.35433961,  36.77351305,
                                    38.4729528 ,  40.48327443,  42.83509356,  45.55902578,
                                    48.68568669,  52.2456919 ,  56.269657  ,  60.78819758,
                                    65.83192926,  71.43146763,  77.61742829,  84.42042684,
                                    91.87107887, 100.        ]).tolist(),
                        np.array([1.00000000e+02, 9.99986302e+01, 9.99890414e+01, 9.99630147e+01,
                                9.99123312e+01, 9.98287718e+01, 9.97041177e+01, 9.95301499e+01,
                                9.92986494e+01, 9.90013972e+01, 9.86301745e+01, 9.81767623e+01,
                                9.76329416e+01, 9.69904934e+01, 9.62411989e+01, 9.53768390e+01,
                                9.43891948e+01, 9.32700474e+01, 9.20111778e+01, 9.06043670e+01,
                                8.90413961e+01, 8.73140462e+01, 8.54140982e+01, 8.33333333e+01,
                                8.10676420e+01, 7.86293526e+01, 7.60349032e+01, 7.33007315e+01,
                                7.04432755e+01, 6.74789732e+01, 6.44242623e+01, 6.12955809e+01,
                                5.81093669e+01, 5.48820580e+01, 5.16300923e+01, 4.83699077e+01,
                                4.51179420e+01, 4.18906331e+01, 3.87044191e+01, 3.55757377e+01,
                                3.25210268e+01, 2.95567245e+01, 2.66992685e+01, 2.39650968e+01,
                                2.13706474e+01, 1.89323580e+01, 1.66666667e+01, 1.45859018e+01,
                                1.26859538e+01, 1.09586039e+01, 9.39563300e+00, 7.98882222e+00,
                                6.72995260e+00, 5.61080518e+00, 4.62316101e+00, 3.75880113e+00,
                                3.00950659e+00, 2.36705844e+00, 1.82323772e+00, 1.36982548e+00,
                                9.98602778e-01, 7.01350648e-01, 4.69850141e-01, 2.95882305e-01,
                                1.71228186e-01, 8.76688310e-02, 3.69852881e-02, 1.09586039e-02,
                                1.36982548e-03, 0.00000000e+00]).tolist()]
        bspline_path = [bspline_path[0].tolist(), bspline_path[1].tolist()]

        # to check if they are equal, we will round to 2dp and compare.
        bspline_path_rounded = [round(x_i) for x in bspline_path for x_i in x]
        expected_path_rounded = [round(x_i) for x in expected_path for x_i in x]

        self.assertCountEqual(bspline_path_rounded, expected_path_rounded)
    
    @unittest.mock.patch('IPPMPlotter.np')
    def test_make_bspline_paths(self, mock_np):

        def linspace_side_effect(start, end, step, endpoint):
            return np.linspace(start, end, step, endpoint=endpoint)
        def array_side_effect(li):
            return np.array(li)
        
        mock_np.linspace.side_effect = linspace_side_effect
        mock_np.array.side_effect = array_side_effect

        # first one is top left to bot right, second is bot left to top right, last is bot left to slightly less left (null edge)
        hexel_coordinate_pairs = [[(0, 100), (100, 0)], [(100, 0), (0, 100)], [(0, 0), (20, 0)]]
        plotter = IPPMPlotter()
        bspline_paths = plotter._make_bspline_paths(hexel_coordinate_pairs)
        expected_paths = []
        for pair in hexel_coordinate_pairs:
            # assuming that _make_bspline_ctr_points works and _make_bspline_path.
            # can do this because we have tests above to verify it.
            if pair == [(0, 0), (20, 0)]:
                expected_paths.append([np.linspace(0, 20, 100, endpoint=True), np.array([0] * 100)])
            else:
                ctr_points = plotter._make_bspline_ctr_points(pair)
                expected_paths.append(plotter._make_bspline_path(ctr_points))
        
        bspline_paths_rounded = [round(x_ii, 2) for x in bspline_paths for x_i in x for x_ii in x_i]
        expected_paths_rounded = [round(x_ii, 2) for x in expected_paths for x_i in x for x_ii in x_i]

        self.assertCountEqual(bspline_paths_rounded, expected_paths_rounded)


    
if __name__ == '__main__':
    unittest.main()
