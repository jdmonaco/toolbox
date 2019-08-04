"""
Source repository information.
"""

import subprocess


class NotARepoError(Exception):
    pass


def git_revision(srcdir, short=False):
    """
    Commit hash for the repository.
    """
    if srcdir is None:
        return None

    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    if not short:
        cmd.remove('--short')

    try:
        output = subprocess.check_output(cmd, cwd=srcdir)
    except subprocess.CalledProcessError:
        raise NotARepoError(srcdir)
    else:
        rev = output.decode().strip()

    return rev
