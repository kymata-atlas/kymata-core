import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from matplotlib import pyplot
from matplotlib.colors import ListedColormap
import os # Import os for SUBJECTS_DIR

# Assuming brain.py is in the same directory or accessible via PYTHONPATH
from kymata.plot.brain import _hexel_minimap_data, _get_colormap_for_cortical_minimap, plot_minimap_hexel

# Mocking necessary Kymata entities for testing
class MockExpressionPoint:
    def __init__(self, channel, transform, logp_value, latency=0.0):
        self.channel = channel
        self.transform = transform
        self.logp_value = logp_value
        self.latency = latency

class MockHexelExpressionSet:
    def __init__(self, hexels_left, hexels_right, transforms, best_transforms_left, best_transforms_right):
        self.hexels_left = np.array(hexels_left)
        self.hexels_right = np.array(hexels_right)
        self.transforms = transforms
        self._best_transforms_left = best_transforms_left
        self._best_transforms_right = best_transforms_right

    def best_transforms(self):
        return self._best_transforms_left, self._best_transforms_right

@pytest.fixture
def mock_expression_set():
    hexels_left = ['lh.hexel_0', 'lh.hexel_1', 'lh.hexel_2', 'lh.hexel_3']
    hexels_right = ['rh.hexel_0', 'rh.hexel_1', 'rh.hexel_2', 'rh.hexel_3']
    transforms = ['transform_A', 'transform_B', 'transform_C']

    # Example best_transforms data
    # These logp_values are intentionally low (very significant)
    best_transforms_left = [
        MockExpressionPoint('lh.hexel_0', 'transform_A', -10.0),
        MockExpressionPoint('lh.hexel_1', 'transform_B', -5.0),
        MockExpressionPoint('lh.hexel_2', 'transform_A', -8.0),
    ]
    best_transforms_right = [
        MockExpressionPoint('rh.hexel_0', 'transform_A', -12.0),
        MockExpressionPoint('rh.hexel_2', 'transform_C', -6.0),
        MockExpressionPoint('rh.hexel_3', 'transform_B', -9.0),
    ]
    return MockHexelExpressionSet(hexels_left, hexels_right, transforms, best_transforms_left, best_transforms_right)

@pytest.fixture
def mock_axes():
    return MagicMock(spec=pyplot.Axes)

class TestHexelMinimapData:
    def test_basic_data_generation(self, mock_expression_set):
        alpha_logp = -7.0 # Only logp_value < -7.0 should pass
        show_transforms = ['transform_A', 'transform_B']
        value_lookup = {'transform_A': 1, 'transform_B': 2, 'transform_C': 3}

        data_left, data_right = _hexel_minimap_data(mock_expression_set, alpha_logp, show_transforms, value_lookup)

        # lh.hexel_0 (A, -10.0) -> significant for A
        # lh.hexel_1 (B, -5.0) -> not significant (as -5.0 is not < -7.0)
        # lh.hexel_2 (A, -8.0) -> significant for A
        # rh.hexel_0 (A, -12.0) -> significant for A
        # rh.hexel_2 (C, -6.0) -> not significant (as -6.0 is not < -7.0)
        # rh.hexel_3 (B, -9.0) -> significant for B

        assert np.all(data_left == np.array([1, 0, 1, 0]))
        assert np.all(data_right == np.array([1, 0, 0, 2]))

    def test_empty_show_transforms(self, mock_expression_set):
        alpha_logp = -7.0
        show_transforms = []
        value_lookup = {'transform_A': 1, 'transform_B': 2}

        data_left, data_right = _hexel_minimap_data(mock_expression_set, alpha_logp, show_transforms, value_lookup)

        assert np.all(data_left == np.array([0, 0, 0, 0]))
        assert np.all(data_right == np.array([0, 0, 0, 0]))

    def test_minimap_latency_range_filtering(self):
        hexels_left = ['lh.hexel_0', 'lh.hexel_1']
        hexels_right = ['rh.hexel_0']
        transforms = ['trans_X']
        best_transforms_left = [
            MockExpressionPoint('lh.hexel_0', 'trans_X', -10.0, latency=0.1),
            MockExpressionPoint('lh.hexel_1', 'trans_X', -10.0, latency=0.5),
        ]
        best_transforms_right = [
            MockExpressionPoint('rh.hexel_0', 'trans_X', -10.0, latency=0.3),
        ]
        mock_es = MockHexelExpressionSet(hexels_left, hexels_right, transforms, best_transforms_left, best_transforms_right)

        alpha_logp = -9.0
        show_transforms = ['trans_X']
        value_lookup = {'trans_X': 1}

        data_left, data_right = _hexel_minimap_data(mock_es, alpha_logp, show_transforms, value_lookup, minimap_latency_range=(0.0, 1.0))
        assert np.all(data_left == np.array([1, 1]))
        assert np.all(data_right == np.array([1]))

        data_left, data_right = _hexel_minimap_data(mock_es, alpha_logp, show_transforms, value_lookup, minimap_latency_range=(0.2, 0.4))
        assert np.all(data_left == np.array([0, 0]))
        assert np.all(data_right == np.array([1]))

        data_left, data_right = _hexel_minimap_data(mock_es, alpha_logp, show_transforms, value_lookup, minimap_latency_range=(0.4, None))
        assert np.all(data_left == np.array([0, 1]))
        assert np.all(data_right == np.array([0]))

        data_left, data_right = _hexel_minimap_data(mock_es, alpha_logp, show_transforms, value_lookup, minimap_latency_range=(None, 0.2))
        assert np.all(data_left == np.array([1, 0]))
        assert np.all(data_right == np.array([0]))


class TestGetColormapForCorticalMinimap:
    def test_colormap_and_lookup_generation(self):
        colors = {
            'transform_X': (1.0, 0.0, 0.0, 1.0),  # Red
            'transform_Y': (0.0, 1.0, 0.0, 1.0),  # Green
            'transform_Z': (0.0, 0.0, 1.0, 1.0),  # Blue
        }
        show_transforms = ['transform_Y', 'transform_X']

        colormap, colormap_value_lookup = _get_colormap_for_cortical_minimap(colors, show_transforms)

        assert isinstance(colormap, ListedColormap)

        num_unique_colors_in_input = len(set(colors.values()))
        assert len(colormap.colors) == num_unique_colors_in_input + 1

        from kymata.plot.color import transparent
        assert colormap.colors[0] == transparent
        assert colormap.colors[1] == (0.0, 0.0, 1.0, 1.0) # Blue
        assert colormap.colors[2] == (0.0, 1.0, 0.0, 1.0) # Green
        assert colormap.colors[3] == (1.0, 0.0, 0.0, 1.0) # Red

        assert colormap_value_lookup['transform_Y'] == pytest.approx(2/3)
        assert colormap_value_lookup['transform_X'] == pytest.approx(1.0)
        assert 'transform_Z' not in colormap_value_lookup


@patch('kymata.datasets.fsaverage.FSAverageDataset')
@patch('kymata.plot.brain.SourceEstimate')
@patch('kymata.plot.brain.pyplot.gcf')
@patch('kymata.plot.brain.pyplot.close')
@patch('kymata.plot.brain.hide_axes')
@patch('kymata.plot.brain._hexel_minimap_data')
@patch('kymata.plot.brain._get_colormap_for_cortical_minimap')
class TestPlotMinimapHexel:
    def test_plot_minimap_hexel_calls_expected_functions(self, mock_get_colormap, mock_hexel_minimap_data, mock_hide_axes, mock_close, mock_gcf, mock_source_estimate, mock_fsaverage_dataset, mock_expression_set, mock_axes):
        mock_fsaverage_dataset.return_value.path = '/mock/fsaverage/path'
        mock_hexel_minimap_data.return_value = (np.array([1, 0, 1]), np.array([2, 0, 2]))

        # Create a specific mock for the screenshot data
        mock_screenshot_data = np.zeros((10, 10))
        mock_colormap_instance = ListedColormap([(0,0,0,0), (1,0,0,1), (0,0,1,1)])
        mock_get_colormap.return_value = (mock_colormap_instance, {'transform_A': 0.5, 'transform_B': 1.0})

        mock_stc_instance = MagicMock()
        mock_stc_instance.plot.return_value.screenshot.return_value = mock_screenshot_data # Return the specific mock object
        mock_source_estimate.return_value = mock_stc_instance

        mock_figure = MagicMock()
        mock_gcf.return_value = mock_figure

        show_transforms = ['transform_A', 'transform_B']
        colors = {'transform_A': 'red', 'transform_B': 'blue'}
        minimap_kwargs = {"smoothing_steps": "sphere", "some_other_kwarg": True}

        plot_minimap_hexel(
            expression_set=mock_expression_set,
            show_transforms=show_transforms,
            lh_minimap_axis=mock_axes,
            rh_minimap_axis=mock_axes,
            view='lateral',
            surface='inflated',
            colors=colors,
            alpha_logp=-7.0,
            minimap_kwargs=minimap_kwargs,
            minimap_latency_range=(0.1, 0.5)
        )

        mock_fsaverage_dataset.assert_called_once_with(download=True)
        assert os.environ["SUBJECTS_DIR"] == '/mock/fsaverage/path'

        mock_get_colormap.assert_called_once_with(colors, show_transforms)
        mock_hexel_minimap_data.assert_called_once_with(
            mock_expression_set,
            alpha_logp=-7.0,
            show_transforms=show_transforms,
            value_lookup={'transform_A': 0.5, 'transform_B': 1.0},
            minimap_latency_range=(0.1, 0.5)
        )

        assert mock_source_estimate.call_count == 1
        args, kwargs = mock_source_estimate.call_args
        assert np.array_equal(kwargs['data'], np.concatenate([np.array([1, 0, 1]), np.array([2, 0, 2])]))
        assert np.array_equal(kwargs['vertices'][0], mock_expression_set.hexels_left)
        assert np.array_equal(kwargs['vertices'][1], mock_expression_set.hexels_right)
        assert kwargs['tmin'] == 0
        assert kwargs['tstep'] == 1

        assert mock_stc_instance.plot.call_count == 2
        expected_plot_kwargs = dict(
            subject="fsaverage",
            surface="inflated",
            views="lateral",
            colormap=mock_colormap_instance,
            smoothing_steps="sphere",
            cortex=dict(colormap="Greys", vmin=-3, vmax=6),
            background="white",
            spacing="ico5",
            time_viewer=False,
            colorbar=False,
            transparent=False,
            clim=dict(
                kind="value",
                lims=(0.0, 0.5, 1.0),
            ),
            some_other_kwarg=True
        )

        lh_call_args, lh_call_kwargs = mock_stc_instance.plot.call_args_list[0]
        assert lh_call_kwargs['hemi'] == "lh"
        for key, value in expected_plot_kwargs.items():
            assert lh_call_kwargs[key] == value

        rh_call_args, rh_call_kwargs = mock_stc_instance.plot.call_args_list[1]
        assert rh_call_kwargs['hemi'] == "rh"
        for key, value in expected_plot_kwargs.items():
            assert rh_call_kwargs[key] == value

        assert mock_axes.imshow.call_count == 2
        # Now assert that imshow was called with the specific mock_screenshot_data object
        mock_axes.imshow.assert_any_call(mock_screenshot_data)