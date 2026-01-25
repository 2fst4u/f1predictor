
import os
import shutil
import pytest
from pathlib import Path
from f1pred.util import ensure_dirs

@pytest.fixture
def temp_cache_dir(tmp_path):
    d = tmp_path / "cache_test"
    yield d
    if d.exists():
        shutil.rmtree(d)

def test_ensure_dirs_permissions(temp_cache_dir):
    """Verify that ensure_dirs creates directories with 0o700 permissions."""
    path_str = str(temp_cache_dir)
    ensure_dirs(path_str)

    assert temp_cache_dir.exists()

    stat = os.stat(path_str)
    mode = stat.st_mode & 0o777

    # Mode should be 0o700
    assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

def test_ensure_dirs_nested(temp_cache_dir):
    """Verify recursive creation and permission on leaf."""
    nested = temp_cache_dir / "subdir" / "leaf"
    path_str = str(nested)

    ensure_dirs(path_str)

    assert nested.exists()

    # Leaf should be 0o700
    stat = os.stat(path_str)
    mode = stat.st_mode & 0o777
    assert mode == 0o700, f"Expected 0o700 for leaf, got {oct(mode)}"

    # Parent (subdir) permissions are system dependent (mkdir default),
    # but that's acceptable as long as leaf is secure.
    # However, if ensure_dirs is called on parents explicitly, they should be secure.
