from kymata.io.config import get_root_dir, get_config_value_with_fallback


def test_import_no_errors():
    get_root_dir({"data_location": "cbu"})


def test_key_exists():
    config = {'key1': 'value1', 'key2': 'value2'}
    result = get_config_value_with_fallback(config, 'key1', 'default_value')
    assert result == 'value1'


def test_key_does_not_exist():
    config = {'key1': 'value1'}
    result = get_config_value_with_fallback(config, 'key2', 'default_value')
    assert result == 'default_value'


def test_fallback_is_used():
    config = {}
    result = get_config_value_with_fallback(config, 'non_existent_key', 'fallback_value')
    assert result == 'fallback_value'


def test_fallback_of_none():
    config = {}
    result = get_config_value_with_fallback(config, 'non_existent_key', None)
    assert result is None
