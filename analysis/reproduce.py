"""Recompute the main statistics reported in the paper from the released data.

Usage: python analysis/reproduce.py   (run from the repository root)
Requires: pandas, scipy
"""
import json
import pandas as pd
from scipy import stats

DATA = "data"


def corr(x, y):
    return [float(round(f(x, y)[0], 3)) for f in (stats.pearsonr, stats.spearmanr, stats.kendalltau)]


# Overall score per system (Table 4): mean over segments of the mean of the three skill scores
print("Overall FALCON score by system")
for pair in ["en-de", "en-es", "en-is"]:
    d = pd.read_json(f"{DATA}/judgments/{pair}.jsonl.gz", lines=True)
    d["s"] = d.scores.apply(lambda l: sum(l) / len(l))
    print(f"\n[{pair}] {len(d)} segment-system pairs, {d.system.nunique()} systems, "
          f"{3 * len(d)} skill scores")
    print(d.groupby("system").s.mean().sort_values(ascending=False).round(2).to_string())

# Agreement with human experts (Table 2a)
r = pd.read_csv(f"{DATA}/human_eval/ratings.csv")
print("\nJudge vs. expert (r, rho, tau_b) over", len(r), "ratings")
for w in ["W1", "W2", "W3"]:
    print(w, corr(r.judge_score, r[w]))
for a, b in [("W1", "W2"), ("W1", "W3"), ("W2", "W3")]:
    print(f"{a}-{b}", corr(r[a], r[b]))

# Correlation with ESA under each context configuration (Table 8, segment level)
c = pd.read_csv(f"{DATA}/context_config/en-es_configurations.csv")
print("\nSegment-level correlation with ESA (r, rho, tau_b), N =", len(c))
for k in ["base", "window4", "no_source", "system_output_target", "current_reference"]:
    s = c[f"score_{k}"].apply(lambda x: sum(json.loads(x)) / 3)
    print(k, corr(s, c.esa))
print("MQM", [float(abs(v)) for v in corr(c.mqm, c.esa)])
