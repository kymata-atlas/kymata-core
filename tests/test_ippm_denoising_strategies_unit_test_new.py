from copy import deepcopy

import numpy as np
import pytest

from kymata.entities.constants import HEMI_LEFT, HEMI_RIGHT
from kymata.entities.expression import ExpressionPoint, HexelExpressionSet
from kymata.ippm.denoising_strategies import DenoisingStrategy, MaxPoolingStrategy
from kymata.math.probability import p_to_logp


"""
    How to set-up fake hexel set data
    =======================================
    
    - Channels are either transforms
    - hexels_lh/rh contain the labels for each hexel/channel
    - latencies contains the labels for latency dimension
    - data_lh/rh actually contains the logp data in the shape (DIM_CHANNEL, DIM_LATENCY, DIM_TRANSFORM)
    - best_transforms retains the lowest logp value over the latencies and transforms per channel
    - Keep them all at 1e-X since the logp value is easy to match (logp 1e-X = X) (but this is unrealistic)
    - the constructor keeps the best transform value per channel over the latencies.
      best_transforms() then chooses the best transform per channel.
"""

simple_test_hexel_set = HexelExpressionSet(
    transforms=["t1", "t2"],
    hexels_lh=range(2),
    hexels_rh=range(2),
    latencies=[5, 10, 15],
    # data shape is DIM_CHANNEL (no of hexels), DIM_LATENCY, DIM_TRANSFORM
    data_lh=p_to_logp(
        # in this hypothetical scenario, let's say that channel 1 Left hemi observes t1, with noise for t2
        np.array([
            [[1e-1, 1e-3], [1e-8, 1e-2], [1e-9, 1e-1]],
            [[1e-3, 1e-4], [1e-5, 1e-2], [1e-3, 1e-2]]
        ])
    ),
    data_rh=p_to_logp(
        # lets say that channel 2 detects t2 with noise for t1
        np.array([
            [[0.1, 0.05], [0.5, 0.3], [0.67, 0.19]],
            [[0.2, 1e-67], [0.1, 1e-15], [0.33, 0.01]]
        ])
    )
)

def test_MaxPoolingStrategy_SimpleHexelSet_DefaultHyperParams():
    test_set = deepcopy(simple_test_hexel_set)
    expected_outcome = HexelExpressionSet(
        transforms=["t1", "t2"],
        hexels_lh=range(2),
        hexels_rh=range(2),
        latencies=[5, 10, 15],
        # data shape is DIM_CHANNEL (no of hexels), DIM_LATENCY, DIM_TRANSFORM
        data_lh=p_to_logp(
            # in this hypothetical scenario, let's say that channel 1 Left hemi observes t1, with noise for t2
            np.array([
                [[1, 1], [1, 1], [1e-9, 1]],
                [[1, 1], [1, 1], [1, 1]]
            ])
        ),
        data_rh=p_to_logp(
            # lets say that channel 2 detects t2 with noise for t1
            np.array([
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1e-67], [1, 1], [1, 1]]
            ])
        )
    )
    strategy = MaxPoolingStrategy()
    test_set = strategy.denoise(test_set)
    print(test_set._data[HEMI_LEFT].data.todense())
    print(expected_outcome._data[HEMI_LEFT].data.todense())
    assert(np.array_equal(expected_outcome._data[HEMI_LEFT].data.todense(), test_set._data[HEMI_LEFT].data.todense()))
    assert(np.array_equal(expected_outcome._data[HEMI_RIGHT].data.todense(), test_set._data[HEMI_RIGHT].data.todense()))