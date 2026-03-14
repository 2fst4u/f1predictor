import fastf1
from f1pred.data.fastf1_backend import init_fastf1

def list_sessions():
    init_fastf1("fastf1_cache")
    for rnd in [1, 2]:
        print(f"\n--- Round {rnd} ---")
        ev = fastf1.get_event(2026, rnd)
        if ev is None:
            print(f"Round {rnd} not found")
            continue
        print(f"Event: {ev.EventName}")
        # Sessions
        for i in range(1, 6):
            try:
                sess = ev.get_session(i)
                print(f"Session {i}: {sess.name}")
                sess.load(telemetry=False, laps=False, weather=False, messages=False)
                if sess.results is not None and not sess.results.empty:
                    print(f"  Results rows: {len(sess.results)}")
                    print(f"  Columns with non-null values: {[c for c in sess.results.columns if sess.results[c].notna().any()]}")
                    if 'Position' in sess.results.columns:
                        print(f"  Position values: {sess.results['Position'].head().tolist()}")
                    if 'ClassifiedPosition' in sess.results.columns:
                        print(f"  ClassifiedPosition values: {sess.results['ClassifiedPosition'].head().tolist()}")
                else:
                    print("  No results")
            except Exception as e:
                print(f"Session {i} failed: {e}")

if __name__ == "__main__":
    list_sessions()
