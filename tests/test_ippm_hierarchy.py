import pytest

from kymata.ippm.hierarchy import CandidateTransformList, TransformHierarchy


@pytest.fixture
def hier() -> TransformHierarchy:
    return {
        'Acc' : ['Vel'],
        'Vel' : [
            'Me1', 
            'Me2', 
            'Me3',
            'Me4',
        ],
        'Me1' : [f'in_{i}' for i in list('12345')],
        'Me2' : [f'in_{i}' for i in list('12345')],
        'Me3' : [f'in_{i}' for i in list('12345')],
        'Me4' : [f'in_{i}' for i in list('12345')],
        'in_1': [],
        'in_2': [],
        'in_3': [],
        'in_4': [],
        'in_5': [],
    }


@pytest.fixture
def ctl(hier) -> CandidateTransformList:
    return CandidateTransformList(hier)


def test_ctl_inputs(ctl):
    assert ctl.inputs == {f'in_{i}' for i in list('12345')}


def test_ctl_transforms(ctl):
    assert ctl.transforms == ({f'in_{i}' for i in list('12345')}
                              | {f'Me{i}' for i in list('1234')}
                              | {'Acc', 'Vel'})


def test_ctl_terminals(ctl):
    assert ctl.terminals == {'Acc'}


def test_ctl_serial_sequence(ctl):
    assert ctl.serial_sequence == (
        {f'in_{i}' for i in list('12345')},
        {f'Me{i}' for i in list('1234')},
        {'Vel'},
        {'Acc'},
    )


@pytest.fixture
def complicated_ctl_example() -> CandidateTransformList:
    r"""
    Not sure if this is realistic, but it should still be treated well

          --- func1 ---       --- func3
         /             \     /
    input ------------- func2
                             \
                              --- func4
    """
    return CandidateTransformList({
        "input": [],
        "func1": ["input"],
        # "func2" also depends on "func1" so can't come in parallel with it, though it's a successor of "input"
        "func2": ["input", "func1"],
        "func3": ["func2"],
        "func4": ["func2"],
    })


def test_ctl_serial_sequence_more_complex(complicated_ctl_example):
    assert complicated_ctl_example.serial_sequence == (
        frozenset({"input"}),
        frozenset({"func1"}),
        frozenset({"func2"}),
        frozenset({"func3", "func4"}),
    )


def test_ctl_subgraph(complicated_ctl_example):
    assert complicated_ctl_example.subgraph(["input", "func1", "func2"]) == CandidateTransformList({
        "input": [],
        "func1": ["input"],
        "func2": ["input", "func1"],
    })
    assert complicated_ctl_example.subgraph(["func1", "func4"]) == CandidateTransformList({
        "func1": [],
        "func4": [],
    })


def test_ctl_connected_components_one(hier):
    ctl = CandidateTransformList(hier)

    assert len(ctl.connected_components) == 1
    assert list(ctl.connected_components)[0] == frozenset(hier.keys())


def test_ctl_connected_components_union(hier):
    # Add a copy of hier to itself
    combined_hier = hier | {
        k+"copy": [t+"copy" for t in v]
        for k, v in hier.items()
    }

    assert len(CandidateTransformList(combined_hier).connected_components) == 2


def test_connected_serial_sequences_one(hier):
    hier = {
        "input": [],
        "func1": ["input"],
        "func2": ["input", "func1"],
        "func3": ["func2"],
        "func4": ["func2"],
    }
    ctl = CandidateTransformList(hier)

    #       --- func1 ---       --- func3
    #      /             \     /
    # input ------------- func2
    #                          \
    #                           --- func4

    assert ctl.connected_serial_sequences == {(
        frozenset({"input"}),
        frozenset({"func1"}),
        frozenset({"func2"}),
        frozenset({"func3", "func4"}),
    )}


def test_connected_serial_sequences_two(hier):
    hier = {
        "input": [],
        "func1": ["input"],
        "func2": ["input", "func1"],
        "func3": ["func2"],
        "func4": ["func2"],
    }
    combined_hier = hier | {
        k + "copy": [t + "copy" for t in v]
        for k, v in hier.items()
    }
    ctl = CandidateTransformList(combined_hier)

    #       --- func1 ---       --- func3
    #      /             \     /
    # input ------------- func2
    #                          \
    #                           --- func4
    #
    #       --- func1_copy ---          --- func3_copy
    #      /                    \     /
    # input_copy ------------- func2_copy
    #                          \
    #                           ----------- func4_copy

    assert ctl.connected_serial_sequences == {(
            frozenset({"input"}),
            frozenset({"func1"}),
            frozenset({"func2"}),
            frozenset({"func3", "func4"}),
        ),
        (
            frozenset({"inputcopy"}),
            frozenset({"func1copy"}),
            frozenset({"func2copy"}),
            frozenset({"func3copy", "func4copy"}),
        )
    }