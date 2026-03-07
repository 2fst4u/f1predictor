import logging
logging.basicConfig(level=logging.INFO)

# Try importing fastf1 FIRST
print("Importing fastf1...")
import fastf1

import requests
import requests_cache

# Simulate util.init_caches
print("Installing requests_cache...")
requests_cache.install_cache('test_cache', backend='memory')

# Simulate fastf1 init
print("Enabling fastf1 cache...")
try:
    fastf1.Cache.enable_cache('fastf1_cache')
    print("Success!")
except Exception as e:
    print(f"FAILED: {e}")
