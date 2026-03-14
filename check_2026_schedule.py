from f1pred.data.jolpica import JolpicaClient
from datetime import datetime, timezone

jc = JolpicaClient("https://api.jolpi.ca/ergast/f1")
schedule = jc.get_season_schedule("2026")
for race in schedule:
    print(f"Round {race['round']}: {race['raceName']} on {race['date']}")
    if "SprintQualifying" in race:
        print(f"  SQ: {race['SprintQualifying']['date']} {race['SprintQualifying']['time']}")

print(f"\nCurrent time: {datetime.now(timezone.utc)}")
