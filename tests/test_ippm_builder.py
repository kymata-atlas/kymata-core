from kymata.ippm.builder import IPPMBuilder
from kymata.ippm.data_tools import IPPMHexel


def test_get_top_level_functions():
    # Graph goes: f4 -> f3 -> f2/f1. I.e., f3 goes into f2 and f1.
    # top level functions should be f2 and f1.
    test_edges = {'f1': ['f3'],
                  'f2': ['f3'],
                  'f3': ['f4'],
                  'f4': []}

    builder = IPPMBuilder()
    top_level_fs = list(builder._get_top_level_functions(test_edges))
    assert set(['f1', 'f2']) == set(top_level_fs)


def test_sort_by_latency():
    test_hexels = {'f1': IPPMHexel('f1')}
    test_hexels['f1'].left_best_pairings = [(-12, 43), (143, 2), (46, 23), (21, 21)]
    test_hexels['f1'].right_best_pairings = [(100, 42), (20, 41), (50, 33)]

    builder = IPPMBuilder()
    sorted = builder._sort_by_latency(test_hexels, 'leftHemisphere', ['f1'])
    sorted = builder._sort_by_latency(sorted, 'rightHemisphere', ['f1'])

    assert [(-12, 43), (21, 21), (46, 23), (143, 2)] == sorted['f1'].left_best_pairings
    assert [(20, 41), (50, 33), (100, 42)] == sorted['f1'].right_best_pairings


def test_build_graph():
    test_hexels = {
        'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2'), 'f3': IPPMHexel('f3')
    }
    test_hexels['f1'].right_best_pairings = [(100, 1e-10)]
    test_hexels['f2'].right_best_pairings = [(90, 1e-15)]
    test_hexels['f3'].right_best_pairings = [(50, 1e-9), (80, 1e-20)]
    test_edges = {'f1': ['f2'],
                  'f2': ['f3'],
                  'f3': ['input'],
                  'input': []}

    builder = IPPMBuilder()
    actual_graph = {
        'f1-0': (100, '#023eff', (100, 1), ['f2-0']),
        'f2-0': (150, '#ff7c00', (90, 0.75), ['f3-1']),
        'f3-0': (90, '#1ac938', (50, 0.5), ['input']),
        'f3-1': (200, '#1ac938', (80, 0.5), ['f3-0']),
        'input': (100, '#e8000b', (0, 0.25), [])
    }
    graph = builder.build_graph(test_hexels, test_edges, ['input'], 'rightHemisphere')
    assert set(actual_graph.keys()) == set(graph.keys())

    for node, val in graph.items():
        assert val.magnitude == actual_graph[node][0]
        assert val.position == actual_graph[node][2]
        assert val.in_edges == actual_graph[node][3]
