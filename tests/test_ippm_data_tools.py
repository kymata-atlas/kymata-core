from kymata.ippm.data_tools import IPPMHexel, build_hexel_dict_from_api_response, causality_violation_score, transform_recall
from collections import namedtuple
import kymata.ippm.data_tools as data_tools
import pytest

Node = namedtuple('Node', 'magnitude position inc_edges')

def test_hexel():
    hexel = IPPMHexel('test', 'test description', 'test commit')
    test_right_pairings = [(20, 10e-3), (50, 0.000012), (611, 0.00053)]
    test_left_pairings = [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    for (left, right) in zip(test_right_pairings, test_left_pairings):
        hexel.add_pairing('rightHemisphere', left)
        hexel.add_pairing('leftHemisphere', right)

    assert hexel.function == 'test'
    assert hexel.description == 'test description'
    assert hexel.github_commit == 'test commit'
    assert hexel.left_best_pairings == [(122, 0.32), (523, 0.00578), (200, 0.0006)]
    assert hexel.right_best_pairings == [(20, 10e-3), (50, 0.000012), (611, 0.00053)]


def test_build_hexel_dict():
    test_dict = {'leftHemisphere': [[2, 1, 0.012, 'left1'], [2, 14, 0.213, 'left1']],
                 'rightHemisphere': [[3, 51, 0.1244, 'left1'], [4, 345, 0.557, 'right1']]}

    hexels = build_hexel_dict_from_api_response(test_dict)

    # check functions are saved correctly
    assert list(hexels.keys()) == ['left1', 'right1']
    # check p value is stored and calculated correctly
    assert hexels['left1'].left_best_pairings == [(1, pow(10, 0.012)), (14, pow(10, 0.213))]
    assert hexels['left1'].right_best_pairings == [(51, pow(10, 0.1244))]

def test_causalityViolation_With_RightHemi_Should_Succeed():
    test_hexels = {
        'f1' : IPPMHexel('f1'),
        'f2' : IPPMHexel('f2'),
        'f3' : IPPMHexel('f3'),
        'f4' : IPPMHexel('f4')
    }
    test_hexels['f1'].right_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels['f2'].right_best_pairings = [(75, 1e-55), (110, 1e-77)]
    test_hexels['f3'].right_best_pairings = [(120, 1e-39)]
    test_hexels['f4'].right_best_pairings = [(100, 1e-19), (150, 1e-75)]
    test_hierarchy = {
        'f4' : ['f3'],
        'f3' : ['f1', 'f2'],
        'f2' : ['f1'],
        'f1' : []
    }

    assert(causality_violation_score(test_hexels, test_hierarchy, 'rightHemisphere', ['f1']) == (0.25, 1, 4))

def test_causalityViolation_With_LeftHemi_Should_Succeed():
    test_hexels = {
        'f1' : IPPMHexel('f1'),
        'f2' : IPPMHexel('f2'),
        'f3' : IPPMHexel('f3'),
        'f4' : IPPMHexel('f4')
    }
    test_hexels['f1'].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels['f2'].left_best_pairings = [(75, 1e-55), (110, 1e-77)]
    test_hexels['f3'].left_best_pairings = [(120, 1e-39)]
    test_hexels['f4'].left_best_pairings = [(100, 1e-19), (150, 1e-75)]
    test_hierarchy = {
        'f4' : ['f3'],
        'f3' : ['f1', 'f2'],
        'f2' : ['f1'],
        'f1' : []
    }
    assert(causality_violation_score(test_hexels, test_hierarchy, 'leftHemisphere', ['f1']) == (0.25, 1, 4))

def test_causalityViolation_With_SingleFunction_Should_Return0():
    test_hexels = {'f1' : IPPMHexel('f1')}
    test_hexels['f1'].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hierarchy = {'f1' : []}

    assert(causality_violation_score(test_hexels, test_hierarchy, 'leftHemisphere', ['f1']) == (0, 0, 0))

def test_causalityViolation_With_SingleEdge_Should_Return0():
    test_hexels = {'f1' : IPPMHexel('f1'), 'f2' : IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = [(50, 1e-50), (100, 1e-25)]
    test_hexels['f2'].left_best_pairings = [(110, 1e-50)]
    test_hierarchy = {'f2' : ['f1'], 'f1' : []}

    assert(causality_violation_score(test_hexels, test_hierarchy, 'leftHemisphere', ['f1']) == (0, 0, 1))

def test_functionRecall_With_NoFuncs_Should_Return0():
    test_hexels = {'f1' : IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = []
    test_hexels['f2'].left_best_pairings = [(10, 1e-1)] # should be > alpha, so not significant
    test_ippm = {}
    funcs = ['f1', 'f2']
    ratio, numer, denom = transform_recall(test_hexels, funcs, test_ippm, 'leftHemisphere')

    assert(ratio == 0)
    assert(numer == 0)
    assert(denom == 0)

def test_functionRecall_With_AllFuncsFound_Should_Return1():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = [(10, 1e-30), (15, 1e-35)]
    test_hexels['f2'].left_best_pairings = [(25, 1e-50), (30, 1e-2)]
    test_ippm = {
        'f1-0': Node(1e-30, 10, []),
        'f1-1': Node(1e-35, 15, ['f1-0']),
        'f2-0': Node(1e-50, 25, ['f1-1'])
    }
    funcs = ['f1', 'f2']
    ratio, numer, denom = transform_recall(test_hexels, funcs, test_ippm, 'leftHemisphere')

    assert(ratio == 1)
    assert(numer == 2)
    assert(denom == 2)

def test_functionRecall_With_InvalidHemiInput_Should_RaiseException():
    with pytest.raises(AssertionError):
        transform_recall({}, [], {}, 'invalidHemisphere')

def test_functionRecall_With_ValidInputRightHemi_Should_ReturnSuccess():
    test_hexels = {'f1': IPPMHexel('f1'), 'f2': IPPMHexel('f2')}
    test_hexels['f1'].left_best_pairings = [(10, 1e-30), (15, 1e-35)]
    test_hexels['f2'].left_best_pairings = [(25, 1e-50), (30, 1e-2)]
    test_ippm = {
        'f1-0': Node(1e-30, 10, []),
        'f1-1': Node(1e-35, 15, ['f1-0'])
    }
    funcs = ['f1', 'f2']
    ratio, numer, denom = transform_recall(test_hexels, funcs, test_ippm, 'leftHemisphere')

    assert(ratio == 1/2)
    assert(numer == 1)
    assert(denom == 2)

def test_Should_convertToPower10_When_validInput():
    hexels = {'f1' : IPPMHexel('f1')}
    hexels['f1'].right_best_pairings = [(10, -50), (20, -10), (30, -20), (40, -3)]
    converted = data_tools.convert_to_power10(hexels)  
    assert converted['f1'].right_best_pairings == [(10, 1e-50), (20, 1e-10), (30, 1e-20), (40, 1e-3)]

def test_Should_removeExcessFuncs_When_validInput():
    hexels = {'f1' : IPPMHexel('f1'), 'f2' : IPPMHexel('f2'), 'f3' : IPPMHexel('f3')}
    to_retain = ['f2']
    filtered = data_tools.remove_excess_funcs(to_retain, hexels)
    assert list(filtered.keys()) == to_retain

def test_Should_copyHemisphere_When_validInput():
    hexels = {'f1' : IPPMHexel('f1')}
    hexels['f1'].right_best_pairings = [(20, 1e-20), (23, 1e-32), (35, 1e-44)]
    hexels['f1'].left_best_pairings = [(10, 1e-20), (21, 1e-55)]
    data_tools.copy_hemisphere(hexels_to=hexels, hexels_from=hexels, hemi_to='rightHemisphere', hemi_from='leftHemisphere', func='f1')
    assert hexels['f1'].right_best_pairings == hexels['f1'].left_best_pairings

