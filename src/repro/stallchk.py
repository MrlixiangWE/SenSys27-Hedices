"""stallchk.py DIR: exit 1 when any result under DIR has a stall (>= 2 samples above 5x the median latency)."""
import glob
import json
import statistics
import sys

bad = []
for f in glob.glob(sys.argv[1] + "/**/hedices_*.json", recursive=True):
    if f.endswith(".net.json"):
        continue
    try:
        lat = json.load(open(f)).get("latency_s") or []
    except Exception:
        continue
    if len(lat) >= 5:
        med = statistics.median(lat)
        if sum(x > 5 * med for x in lat) >= 2:
            bad.append(f)
print("\n".join("stall: " + b for b in bad))
sys.exit(1 if bad else 0)
