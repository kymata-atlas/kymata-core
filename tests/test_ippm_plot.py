import pytest
from numpy import array
from numpy.typing import NDArray

from kymata.entities.expression import SensorExpressionSet
from kymata.ippm.ippm import IPPM
from kymata.math.probability import p_to_logp
from kymata.plot import expression_plot
from kymata.plot.ippm import plot_ippm

sensors = [str(i) for i in range(4)]
transform_a_data: NDArray = array(
    p_to_logp(
        array(
            [
                # 0  1  2  3  latencies
                [1, 1e-20, 1, 1],  # 0
                [1, 1, 1, 1],  # 1
                [1, 1, 1, 1],  # 2
                [1, 1, 1, 1],  # 3 sensors
            ]
        )
    )
)
transform_b_data: NDArray = array(
    p_to_logp(
        array(
            [
                [1, 1, 1, 1],
                [1, 1, 1e-25, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
    )
)
transform_c_data: NDArray = array(
    p_to_logp(
        array(
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1e-25],
            ]
        )
    )
)


def test_ippm_plot_runs():
    es = SensorExpressionSet(
        transforms=["a", "b", "c"],
        sensors=sensors,  # 4
        latencies=[0, .05, .1, .15],
        data=[transform_a_data, transform_b_data, transform_c_data],
    )

    ippm = IPPM(es, {"input": [], "a": ["input"], "b": ["a"], "c": ["a"]})
    plot_ippm(ippm, colors={"input": "grey", "a": "red", "b": "blue", "c": "green"})


def test_ippm_plot_runs_with_causality_violation():
    es = SensorExpressionSet(
        transforms=["a", "b", "c"],
        sensors=sensors,  # 4
        latencies=[0, .05, .1, .15],
        data=[transform_b_data, transform_a_data, transform_c_data],
    )

    ippm = IPPM(es, {"input": [], "a": ["input"], "b": ["a"], "c": ["a"]})
    plot_ippm(ippm, colors={"input": "grey", "a": "red", "b": "blue", "c": "green"})
