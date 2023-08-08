import unittest
import unittest.mock
from data_tools import Hexel
from graph_builder import GraphBuilder
import networkx as nx
import seaborn as sns

class TestGraphBuilder(unittest.TestCase):
    def test_set_node_attributes(self):
        test_hexels = {
                'f1' : Hexel('f1'), 'f2' : Hexel('f2'), 'f3' : Hexel('f3')
            }
        test_hexels['f1'].right_best_pairings = [(100, 1e-10)]
        test_hexels['f2'].right_best_pairings = [(90, 1e-15)]
        test_hexels['f3'].right_best_pairings = [(50, 1e-9), (80, 1e-20)]
        graph = nx.DiGraph()
        # emulate how draw works. Function name - pairing number.
        graph.add_node('f1-0')
        graph.add_node('f2-0')
        graph.add_node('f3-0')
        graph.add_node('f3-1')
        graph.add_edge('f2-0', 'f1-0')
        graph.add_edge('f3-1', 'f3-0')
        graph.add_edge('f3-1', 'f2-0')
        colors = sns.color_palette('bright',3).as_hex()
        f_colors = {f: color for f, color in zip(['f1', 'f2', 'f3'], colors)}

        builder = GraphBuilder()
        node_sizes, node_colors = builder._set_node_attributes(graph, test_hexels, f_colors, 'rightHemisphere')
        actual_node_sizes = [100, 150, 90, 200]
        actual_colors = [f_colors['f1'], f_colors['f2'], f_colors['f3'], f_colors['f3']]

        self.assertEqual(actual_node_sizes, node_sizes)
        self.assertEqual(actual_colors, node_colors)


    def test_get_top_level_functions(self):
        # Graph goes: f4 -> f3 -> f2/f1. I.e., f3 goes into f2 and f1.
        # top level functions should be f2 and f1.
        test_edges = {'f1' : ['f3'],
                      'f2' : ['f3'],
                      'f3' : ['f4'],
                      'f4' : []}
        
        builder = GraphBuilder()
        top_level_fs = list(builder._get_top_level_functions(test_edges))
        self.assertEqual(set(['f1', 'f2']), set(top_level_fs))


    def test_sort_by_latency(self):
        test_hexels = {'f1' : Hexel('f1')}
        test_hexels['f1'].left_best_pairings = [(-12, 43), (143, 2), (46, 23), (21, 21)]
        test_hexels['f1'].right_best_pairings = [(100, 42), (20, 41), (50, 33)]

        builder = GraphBuilder()
        sorted = builder._sort_by_latency(test_hexels, 'leftHemisphere')
        sorted = builder._sort_by_latency(sorted, 'rightHemisphere')

        self.assertEqual([(-12, 43), (21, 21), (46, 23), (143, 2)], sorted['f1'].left_best_pairings)
        self.assertEqual([(20, 41), (50, 33), (100, 42)], sorted['f1'].right_best_pairings)

    def test_draw(self):
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
        builder = GraphBuilder()
        graph = builder.draw(test_hexels, test_edges, ['input'], 'rightHemisphere', 'test')

        actual_nodes = ['f1-0', 'f2-0', 'f3-0', 'f3-1', 'input']
        actual_edges = [('f2-0', 'f1-0'), ('f3-1', 'f2-0'), ('f3-0', 'f3-1'),
                        ('input', 'f3-0')]
        self.assertEqual(set(graph.edges), set(actual_edges))
        self.assertEqual(set(graph.nodes), set(actual_nodes))


if __name__ == '__main__':
    unittest.main()