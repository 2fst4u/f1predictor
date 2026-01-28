
import unittest
from unittest.mock import MagicMock
from f1pred.util import http_get_json

class TestSecurityDoS(unittest.TestCase):
    def test_content_length_check_enforced(self):
        """
        Security Test: Ensure that http_get_json strictly respects the Content-Length header.
        Prevents DoS via resource consumption by failing FAST if the header indicates
        a response larger than the 10MB limit.
        """
        session = MagicMock()
        response = MagicMock()
        session.get.return_value.__enter__.return_value = response

        # Scenario: Content-Length is valid integer but too large (20MB)
        # MAX_SIZE is 10MB in util.py
        response.headers = {"Content-Length": str(20 * 1024 * 1024)}
        response.iter_content.return_value = [b"chunk"]
        response.encoding = "utf-8"

        # Must raise ValueError("Response too large...")
        with self.assertRaises(ValueError) as cm:
            http_get_json(session, "http://example.com")

        self.assertIn("Response too large", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
