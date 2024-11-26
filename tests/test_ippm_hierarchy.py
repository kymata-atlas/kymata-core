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
    assert ctl.serial_sequence == [
        [f'in_{i}' for i in list('12345')],
        [f'Me{i}' for i in list('1234')],
        ['Vel'],
        ['Acc'],
    ]
