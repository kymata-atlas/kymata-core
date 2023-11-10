from kymata.ippm.data_tools import IPPMHexel, build_hexel_dict_from_api_response


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

