from pathlib import Path


def kymata_installed_as_dependency() -> bool:
    """
    Does a rough heuristic attempt to detect if kymata has been installed as a dependency, or if it has been checked out
    directly.

    Returns:
        bool: True iff probably installed as a dependency.
    """
    package_path = Path(__file__)

    # If installed with poetry, expect it to be in a "site-packages" directory.
    return "site-packages" in str(package_path) or "dist-packages" in str(package_path)
