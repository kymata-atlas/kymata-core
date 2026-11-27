from logging import getLogger

import git


_logger = getLogger(__file__)


def get_current_git_commit_hash(verbose: bool = False) -> str | None:
    """Gets the current git commit hash. Returns None on a git error (e.g. not inside a repository)."""
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.GitError as ex:
        if verbose:
            _logger.warning(f"Could not get commit hash: {ex!s}")
        return None
