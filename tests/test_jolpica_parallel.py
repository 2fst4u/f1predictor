import unittest
import time
from unittest.mock import MagicMock, patch, call
from f1pred.data.jolpica import JolpicaClient

class TestJolpicaParallel(unittest.TestCase):
    def test_fetch_paginated_parallel_logic(self):
        """Test that pagination logic spawns correct tasks and collects results."""
        client = JolpicaClient("http://mock", timeout=1)

        def side_effect(path, params=None, **kwargs):
            offset = params.get("offset", 0)
            limit = params.get("limit", 100)

            return {
                "MRData": {
                    "total": "25",
                    "limit": str(limit),
                    "offset": str(offset),
                    "data": f"content_{offset}"
                }
            }

        client._get = MagicMock(side_effect=side_effect)

        results = client._fetch_paginated_parallel("test/path", limit=10)

        # Verify results
        self.assertEqual(len(results), 3)
        offsets = [int(r["offset"]) for r in results]
        self.assertEqual(offsets, [0, 10, 20])

        # Verify _get calls
        self.assertEqual(client._get.call_count, 3)

    def test_fetch_paginated_parallel_ordering(self):
        """Test that results are collected in offset order even if they complete out of order."""
        client = JolpicaClient("http://mock", timeout=1)

        # Scenario: Total 30 items, Limit 10.
        # Pages: 0 (seq), 10, 20 (parallel)

        def side_effect(path, params=None, **kwargs):
            offset = params.get("offset", 0)

            # Simulate delay for offset 10 to ensure offset 20 finishes "first" in thread pool
            # (Though thread execution order is not guaranteed, this helps simulate race conditions)
            if offset == 10:
                time.sleep(0.05)

            return {
                "MRData": {
                    "total": "30",
                    "limit": "10",
                    "offset": str(offset),
                    "data": f"page_{offset}"
                }
            }

        client._get = MagicMock(side_effect=side_effect)

        # Use more workers to ensure parallelism
        results = client._fetch_paginated_parallel("test/path", limit=10, max_workers=5)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["offset"], "0")
        self.assertEqual(results[1]["offset"], "10")
        self.assertEqual(results[2]["offset"], "20")

    def test_fetch_paginated_single_page(self):
        """Test case where total < limit."""
        client = JolpicaClient("http://mock", timeout=1)

        def side_effect(path, params=None, **kwargs):
            return {
                "MRData": {
                    "total": "5",
                    "limit": "10",
                    "offset": "0",
                    "data": "single_page"
                }
            }

        client._get = MagicMock(side_effect=side_effect)

        results = client._fetch_paginated_parallel("test/path", limit=10)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["offset"], "0")
        self.assertEqual(client._get.call_count, 1)

if __name__ == '__main__':
    unittest.main()
