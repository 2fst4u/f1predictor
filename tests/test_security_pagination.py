import unittest
from unittest.mock import MagicMock, patch
from f1pred.data.jolpica import JolpicaClient

class TestSecurityPagination(unittest.TestCase):
    def test_pagination_dos_protection(self):
        """Verify that pagination loops are bounded to prevent DoS."""
        client = JolpicaClient("http://test.local")

        # Simulate a malicious/buggy API returning a huge total
        # limit=100. total=100000 -> 1000 iterations.
        # We expect it to stop at MAX_PAGINATION_PAGES (e.g., 20)

        huge_total = "100000"
        mock_response = {
            "MRData": {
                "total": huge_total,
                "RaceTable": {
                    "Races": [{"round": "1", "Results": [{"position": "1"}]}]
                }
            }
        }

        # We mock _get to return this static response every time
        # The loop increments offset locally, so it keeps requesting.
        client._get = MagicMock(return_value=mock_response)

        # We also need to mock _validate_season to avoid API calls or validation errors
        client._validate_season = MagicMock(return_value="2023")

        # Run the method
        # If the fix works, this returns quickly.
        # If not, it runs 1000 times (which is still fast in mock, but strictly > 20).
        results = client.get_season_race_results("2023")

        # Verify call count
        # We assume MAX_PAGINATION_PAGES = 20
        # Check if the constant exists on the class or module, or just hardcode the expected value
        MAX_PAGES = getattr(JolpicaClient, "MAX_PAGINATION_PAGES", 20)

        # It should be exactly MAX_PAGES or MAX_PAGES + 1 depending on implementation
        # (check at start or end of loop).
        # We want to ensure it is significantly less than 1000.
        self.assertLess(client._get.call_count, 100, "Pagination loop did not terminate early (potential DoS)")
        self.assertLessEqual(client._get.call_count, MAX_PAGES + 1, "Pagination loop exceeded defined limit")

if __name__ == "__main__":
    unittest.main()
