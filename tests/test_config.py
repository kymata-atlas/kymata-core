from kymata.io.config import get_root_dir


def test_import_no_errors():
    get_root_dir({"data_location": "cbu"})
