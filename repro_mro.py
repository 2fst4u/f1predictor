import requests_cache
import logging

logging.basicConfig(level=logging.INFO)

# Simulate util.init_caches
print("Installing requests_cache...")
requests_cache.install_cache('test_cache', backend='memory')

# Simulate fastf1 init
print("Importing fastf1...")
try:
    import fastf1
    print("Enabling fastf1 cache...")
    fastf1.Cache.enable_cache('fastf1_cache')
    print("Success!")
except Exception as e:
    print(f"FAILED: {e}")
