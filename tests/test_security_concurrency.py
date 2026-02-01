
import threading
import os
import time
import pytest
from f1pred.util import ensure_dirs

def worker(path):
    # Simulate work that calls ensure_dirs
    ensure_dirs(path)
    # Adding a small sleep to increase the window of race condition overlap
    time.sleep(0.001)

def test_umask_race_condition(tmp_path):
    """
    Verify that concurrent calls to ensure_dirs do not corrupt the global umask.

    Vulnerability: ensure_dirs changes os.umask() temporarily. In a multithreaded context,
    calls can overlap, causing the 'restored' umask to be the temporary restrictive one (0o77),
    permanently altering the process state.
    """
    # 1. Set a known initial umask (e.g. 022, standard for many systems)
    initial_umask = 0o022
    old = os.umask(initial_umask)

    # Verify we set it correctly (umask returns PREVIOUS, so call again to read current, then restore)
    current = os.umask(initial_umask)
    assert current == initial_umask, f"Failed to set initial umask, got {oct(current)}"

    try:
        # 2. Spawn threads
        threads = []
        n_threads = 50
        for i in range(n_threads):
            # Use unique paths to avoid filesystem errors, we only care about umask state
            p = str(tmp_path / f"dir_{i}")
            t = threading.Thread(target=worker, args=(p,))
            threads.append(t)

        # 3. Start all
        for t in threads:
            t.start()

        # 4. Join all
        for t in threads:
            t.join()

        # 5. Check umask
        final_umask = os.umask(initial_umask) # Read and reset

        # If the race condition hit, final_umask will likely be 0o077
        # We assert it should match initial_umask
        assert final_umask == initial_umask, f"Umask was corrupted! Expected {oct(initial_umask)}, got {oct(final_umask)}"

    finally:
        # Always restore umask to avoid contaminating other tests if this one fails
        os.umask(initial_umask)
