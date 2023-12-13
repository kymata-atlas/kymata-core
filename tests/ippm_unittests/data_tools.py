import unittest
import unittest.mock as mock

from kymata.ippm import data_tools


class TestDataTools(unittest.TestCase):

    @mock.patch('data_tools.json')
    @mock.patch('data_tools.requests')
    def test_fetch_data(self, mock_requests, mock_json):
        # set up mock
        test_dict = {'leftHemisphere' : [[2, 1, 0.012, 'left1'], [2, 14, 0.213, 'left1']],
                     'rightHemisphere' : [[3, 51, 0.1244, 'left1'], [4, 345, 0.557, 'right1']]}

        mock_requests.get.return_value = mock_requests # we dont care about return since we mock json and return test_dict by mocking the json
        mock_requests.text.return_value = 'testing'
        mock_json.loads.return_value = test_dict 

        hexels = data_tools.fetch_data('testing')
        self.assertEqual(list(hexels.keys()), ['left1', 'right1']) # check functions are saved correctly
        # check p value is stored and calculated correctly
        self.assertEqual(hexels['left1'].left_best_pairings, [(1, pow(10, 0.012)), (14, pow(10, 0.213))])
        self.assertEqual(hexels['left1'].right_best_pairings, [(51, pow(10, 0.1244))])
        

if __name__ == '__main__':
    unittest.main()
