import unittest.mock
from kymata.ippm.data_tools import Hexel
from kymata.ippm.builder import IPPMBuilder


class TestIPPMBuilder(unittest.TestCase):
    def test_get_top_level_functions(self):
        # Graph goes: f4 -> f3 -> f2/f1. I.e., f3 goes into f2 and f1.
        # top level functions should be f2 and f1.
        test_edges = {'f1' : ['f3'],
                      'f2' : ['f3'],
                      'f3' : ['f4'],
                      'f4' : []}
        
        builder = IPPMBuilder()
        top_level_fs = list(builder._get_top_level_functions(test_edges))
        self.assertEqual(set(['f1', 'f2']), set(top_level_fs))


    def test_sort_by_latency(self):
        test_hexels = {'f1' : Hexel('f1')}
        test_hexels['f1'].left_best_pairings = [(-12, 43), (143, 2), (46, 23), (21, 21)]
        test_hexels['f1'].right_best_pairings = [(100, 42), (20, 41), (50, 33)]

        builder = IPPMBuilder()
        sorted = builder._sort_by_latency(test_hexels, 'leftHemisphere')
        sorted = builder._sort_by_latency(sorted, 'rightHemisphere')

        self.assertEqual([(-12, 43), (21, 21), (46, 23), (143, 2)], sorted['f1'].left_best_pairings)
        self.assertEqual([(20, 41), (50, 33), (100, 42)], sorted['f1'].right_best_pairings)

    def test_build_graph(self):
        test_hexels = {
                'f1' : Hexel('f1'), 'f2' : Hexel('f2'), 'f3' : Hexel('f3')
            }
        test_hexels['f1'].right_best_pairings = [(100, 1e-10)]
        test_hexels['f2'].right_best_pairings = [(90, 1e-15)]
        test_hexels['f3'].right_best_pairings = [(50, 1e-9), (80, 1e-20)]
        test_edges = {'f1' : ['f2'],
                      'f2' : ['f3'],
                      'f3' : ['input'],
                      'input' : []}
        
        builder = IPPMBuilder()
        actual_graph = {
                'f1-0' : (100, '#023eff', (100, 1), ['f2-0']),
                'f2-0' : (150, '#ff7c00', (90, 0.75), ['f3-1']),
                'f3-0' : (90, '#1ac938', (50, 0.5), ['input']),
                'f3-1' : (200, '#1ac938', (80, 0.5), ['f3-0']),
                'input' : (100, '#e8000b', (0, 0.25), [])
            }
        actual_colors = {
                'f1' : '#023eff',
                'f2' : '#ff7c00',
                'f3' : '#1ac938',
                'input' : '#e8000b'
        }
        graph, colors = builder.build_graph(test_hexels, test_edges, ['input'], 'rightHemisphere')
        self.assertEqual(set(actual_graph.keys()), set(graph.keys()))
        self.assertEqual(set(actual_colors.keys()), set(colors.keys()))

        for node, val in graph.items():
            self.assertEqual(val.magnitude, actual_graph[node][0])
            self.assertEqual(val.color, actual_graph[node][1])
            self.assertEqual(val.position, actual_graph[node][2])
            self.assertEqual(val.in_edges, actual_graph[node][3])

if __name__ == '__main__':
    unittest.main()