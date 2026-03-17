## 2024-XX-XX - CI/local coverage discrepancies
**Learning:** Local coverage calculations can occasionally report higher percentages than the GitHub CI runner due to environment discrepancies. When ratcheting the coverage threshold (`fail_under`), align with the CI runner's metric to prevent pipeline failures.
**Action:** When bumping the `fail_under` threshold, bump it by exactly the delta amount of the new tests instead of directly setting it to the local coverage total, or use the local delta to add to the previously set threshold.
