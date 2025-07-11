import pytest

from kymata.ippm.hierarchy import CandidateTransformList, TransformHierarchy


@pytest.fixture
def hier() -> TransformHierarchy:
    return {
        'Acc':  {'Vel'},
        'Vel':  {
            'Me1', 
            'Me2', 
            'Me3',
            'Me4',
        },
        'Me1':  {f'in_{i}' for i in list('12345')},
        'Me2':  {f'in_{i}' for i in list('12345')},
        'Me3':  {f'in_{i}' for i in list('12345')},
        'Me4':  {f'in_{i}' for i in list('12345')},
        'in_1': set(),
        'in_2': set(),
        'in_3': set(),
        'in_4': set(),
        'in_5': set(),
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
    assert ctl.serial_sequence == [
        {f'in_{i}' for i in list('12345')},
        {f'Me{i}' for i in list('1234')},
        {'Vel'},
        {'Acc'},
    ]


def test_ctl_serial_sequence_more_complex():
    # Not sure if this is realistic, but it should still be treated well
    ctl = CandidateTransformList({
        "input": set(),
        "func1": {"input"},
        # "func2" also depends on "func1" so can't come in parallel with it, though it's a successor of "input"
        "func2": {"input", "func1"},
        "func3": {"func2"},
        "func4": {"func2"},
    })

    #       --- func1 ---       --- func3
    #      /             \     /
    # input ------------- func2
    #                          \
    #                           --- func4

    assert ctl.serial_sequence == [
        {"input"},
        {"func1"},
        {"func2"},
        {"func3", "func4"},
    ]


def test_hierarchy_recoverable(hier):
    assert CandidateTransformList(hier).hierarchy == hier


def test_merge_ct_fails_with_overlapping_transforms():
    left = CandidateTransformList({
        "input": set(),
        "func1": {"input"},
        "func2": {"input", "func1"},
    })
    right = CandidateTransformList({
        "input": set(),
        "func1": {"input"},
        "func3": {"input", "func1"},
    })
    with pytest.raises(NotImplementedError):
        CandidateTransformList.merge(left, right)


def test_merge_ctl_with_overlapping_inputs():
    left = CandidateTransformList({
        "input": set(),
        "left_func1": {"input"},
        "left_func2": {"input", "left_func1"},
        "left_func3": {"left_func2"},
        "left_func4": {"left_func2"},
    })
    right = CandidateTransformList({
        "input": set(),
        "right_func1": {"input"},
        "right_func2": {"input", "right_func1"},
        "right_func3": {"right_func2"},
        "right_func4": {"right_func2"},
    })
    combined = CandidateTransformList.merge(left, right)
    assert combined == CandidateTransformList({
        "input": set(),
        "left_func1": {"input"},
        "left_func2": {"input", "left_func1"},
        "left_func3": {"left_func2"},
        "left_func4": {"left_func2"},
        "right_func1": {"input"},
        "right_func2": {"input", "right_func1"},
        "right_func3": {"right_func2"},
        "right_func4": {"right_func2"},
    })


def test_merge_ctl_with_disjoint_inputs():
    left = CandidateTransformList({
        "left_input": set(),
        "left_func1": {"left_input"},
        "left_func2": {"left_input", "left_func1"},
        "left_func3": {"left_func2"},
        "left_func4": {"left_func2"},
    })
    right = CandidateTransformList({
        "right_input": set(),
        "right_func1": {"right_input"},
        "right_func2": {"right_input", "right_func1"},
        "right_func3": {"right_func2"},
        "right_func4": {"right_func2"},
    })
    combined = CandidateTransformList.merge(left, right)
    assert combined.inputs == {"left_input", "right_input"}
    assert combined == CandidateTransformList({
        "left_input": set(),
        "right_input": set(),
        "left_func1": {"left_input"},
        "left_func2": {"left_input", "left_func1"},
        "left_func3": {"left_func2"},
        "left_func4": {"left_func2"},
        "right_func1": {"right_input"},
        "right_func2": {"right_input", "right_func1"},
        "right_func3": {"right_func2"},
        "right_func4": {"right_func2"},
    })
