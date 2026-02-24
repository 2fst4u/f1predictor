
import unittest
from unittest.mock import MagicMock
from f1pred.data.jolpica import JolpicaClient

class TestSecurityJolpicaValidation(unittest.TestCase):
    def setUp(self):
        self.client = JolpicaClient("http://mock-api.com")

    def test_validate_driver_id_valid(self):
        """Test that valid driver IDs are accepted."""
        valid_ids = [
            "hamilton",
            "max_verstappen",
            "driver-1",
            "driver.name",
            "123",
            "driver_name-123.456"
        ]
        for did in valid_ids:
            self.assertEqual(self.client._validate_driver_id(did), did)

    def test_validate_driver_id_invalid_chars(self):
        """Test that driver IDs with invalid characters are rejected."""
        invalid_ids = [
            "driver/1",
            "driver<script>",
            "../../etc/passwd",
            "driver space",
            "driver@domain",
            "driver!",
            "driver$"
        ]
        for did in invalid_ids:
            with self.assertRaises(ValueError, msg=f"Should raise ValueError for {did}"):
                self.client._validate_driver_id(did)

    def test_validate_driver_id_length(self):
        """Test that driver IDs exceeding the length limit are rejected."""
        # Max length is 64
        valid_long = "a" * 64
        self.assertEqual(self.client._validate_driver_id(valid_long), valid_long)

        invalid_long = "a" * 65
        with self.assertRaises(ValueError):
            self.client._validate_driver_id(invalid_long)

    def test_get_season_entry_list_handles_validation_failure(self):
        """Test that get_season_entry_list handles validation failure gracefully."""
        # Mock get_drivers_for_season
        self.client.get_drivers_for_season = MagicMock(return_value=[
            {"driverId": "valid_driver"},
            {"driverId": "invalid/driver"},  # Should fail validation
            {"driverId": "another_valid"}
        ])

        # Mock _get for the team fetching
        def mock_get(url, **kwargs):
            if "valid_driver" in url:
                return {"MRData": {"ConstructorTable": {"Constructors": [{"constructorId": "team1"}]}}}
            if "another_valid" in url:
                return {"MRData": {"ConstructorTable": {"Constructors": [{"constructorId": "team2"}]}}}
            if "invalid/driver" in url:
                # This should NOT be reached if validation works
                raise RuntimeError("Validation failed to stop invalid driverId request")
            return {}

        self.client._get = MagicMock(side_effect=mock_get)

        # Run the method
        entries = self.client.get_season_entry_list("2024")

        # We expect 3 entries (2 valid + 1 invalid but caught)
        self.assertEqual(len(entries), 3)

        # Verify calls
        # Ensure mock_get was NOT called with the invalid URL
        # We can inspect call args
        for call_args in self.client._get.call_args_list:
            url = call_args[0][0]
            self.assertNotIn("invalid/driver", url)

        # Verify the invalid driver entry has empty constructor
        invalid_entry = next(e for e in entries if e["Driver"]["driverId"] == "invalid/driver")
        self.assertEqual(invalid_entry["Constructor"], {})

if __name__ == "__main__":
    unittest.main()
