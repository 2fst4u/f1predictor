
import unittest
import logging
from f1pred.util import sanitize_for_console
from f1pred.data.jolpica import JolpicaClient
from f1pred.data.open_meteo import OpenMeteoClient

class TestSecurityInputLimits(unittest.TestCase):
    def test_sanitize_truncation(self):
        """
        Security Test: Ensure sanitize_for_console truncates extremely long inputs
        to prevent Log Flooding and Terminal DoS.
        """
        # Create a massive string (e.g. 10KB)
        long_input = "A" * 10000
        sanitized = sanitize_for_console(long_input)

        # Expect truncation to reasonable limit (e.g. 1024)
        # We test for <= 1024 + minimal overhead (suffix)
        # 1024 + len("...[truncated]") = 1024 + 14 = 1038
        self.assertLessEqual(len(sanitized), 1050)
        self.assertTrue(sanitized.startswith("AAAA"))
        self.assertTrue(sanitized.endswith("[truncated]"))

    def test_jolpica_round_limit(self):
        """
        Security Test: Ensure JolpicaClient rejects excessively long round strings
        to prevent upstream DoS / URL length issues.
        """
        jc = JolpicaClient("http://mock")
        long_round = "1" * 1000  # 1000 digits

        # Should raise ValueError immediately due to length, before isdigit checks
        # or as part of validation
        with self.assertRaises(ValueError):
            jc._validate_round(long_round)

    def test_openmeteo_timezone_limit(self):
        """
        Security Test: Ensure OpenMeteoClient limits timezone string length.
        """
        om = OpenMeteoClient("http://mock", "http://mock", "http://mock", "http://mock", "http://mock")
        long_tz = "Europe/London" + ("/" * 1000)

        # Should return "UTC" fallback or raise, but definitely not return the massive string
        result = om._validate_timezone(long_tz)
        self.assertEqual(result, "UTC")

if __name__ == "__main__":
    unittest.main()
