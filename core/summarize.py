import numpy as np
from nav_vlm.core.types import SceneSummary, Hazard

def summarize(depth_m, masks, danger_thr=1.5):
    zones = {}
    for name, m in masks.items():
        vals = depth_m[m]
        if vals.size == 0:
            zones[name] = float("inf"); continue
        k = max(1, int(0.2*vals.size))
        zones[name] = float(np.mean(np.sort(vals)[:k]))

    hazards = [Hazard("obstacle", round(d,2), name)
               for name,d in zones.items() if d < danger_thr]
    best = max(zones.items(), key=lambda kv: kv[1])[0] if zones else "center"

    return SceneSummary(hazards,
                        {k:(None if np.isinf(v) else round(v,2)) for k,v in zones.items()},
                        recommendation=best)
