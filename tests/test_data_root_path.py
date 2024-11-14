import os
from pathlib import Path
from unittest.mock import patch
from pytest import fixture, raises

from kymata.datasets.data_root import data_root_path


def clear_env_var():
    """Helper function to clear the environment variable"""
    os.environ.pop('KYMATA_DATA_ROOT', None)


@fixture(autouse=True)
def reset_env():
    """Reset environment before and after each test"""
    clear_env_var()
    yield
    clear_env_var()


def test_data_root_provided():
    data_root = '/valid/path/to/data'

    with patch('pathlib.Path.exists', return_value=True), \
            patch('pathlib.Path.is_dir', return_value=True):
        result = data_root_path(data_root)
        assert result == Path(data_root)


def test_data_root_env_variable():
    os.environ['KYMATA_DATA_ROOT'] = '/valid/env/path'

    with patch('pathlib.Path.exists', return_value=True), \
            patch('pathlib.Path.is_dir', return_value=True):
        result = data_root_path()
        assert result == Path(os.environ['KYMATA_DATA_ROOT'])


def test_data_root_invalid_directory():
    data_root = '/invalid/path/to/data'

    with patch('pathlib.Path.exists', return_value=False):
        with raises(FileNotFoundError):
            data_root_path(data_root)

    with patch('pathlib.Path.exists', return_value=True), \
            patch('pathlib.Path.is_dir', return_value=False):
        with raises(NotADirectoryError):
            data_root_path(data_root)
