"""
Phase 5 - predicted-vs-actual tracking for the knockout rounds.

Locks in the model's pre-match call for every knockout tie, then scores rolling
accuracy (and a Brier score for calibration) as real results arrive. The rule
that keeps it honest: a prediction is written to the log *once* and never
recomputed, so each row always reflects what the model said before the match.

Workflow per round:
  1. Add the round's confirmed ties to FIXTURES, run this script -> predictions
     are logged using the current (pre-round) ratings.
  2. As results land, add the score to results_2026_manual.csv (updates ratings)
     and fill actual_advance / actual_score in the tracking CSV, then re-run to
     score accuracy.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, "src")
from elo import load_matches, run_elo  # noqa: E402
from simulate import (  # noqa: E402
    advance_prob,
    calibrate,
    expected_goals,
    replay_history,
    shootout_prob,
)

LOG_PATH = "data/processed/knockout_tracking.csv"

# Knockout ties, added round by round as the draw confirms them.
FIXTURES = [
    ("R16", "Paraguay", "France"),
    ("R16", "Canada", "Morocco"),
    ("R16", "Portugal", "Spain"),
    ("R16", "United States", "Belgium"),
    ("R16", "Brazil", "Norway"),
    ("R16", "Mexico", "England"),
    ("R16", "Argentina", "Egypt"),
    ("R16", "Switzerland", "Colombia"),
]

COLUMNS = ["round", "home", "away", "pred_advance", "pred_prob", "exp_score",
           "actual_advance", "actual_score", "decided_by", "hit"]


def predict(home: str, away: str, ratings: dict, params: dict) -> tuple:
    """Return (predicted advancer, its advance-prob %, expected scoreline)."""
    eh, ea = ratings[home], ratings[away]
    lh, la = expected_goals(eh, ea, neutral=True, params=params)
    p_home = advance_prob(lh, la, pen_home=shootout_prob(eh, ea))
    exp = f"{lh:.1f}-{la:.1f}"
    if p_home >= 0.5:
        return home, round(100 * p_home, 1), exp
    return away, round(100 * (1 - p_home), 1), exp


def main() -> None:
    matches = load_matches()
    ratings = run_elo(matches).set_index("team")["rating"].to_dict()
    params = calibrate(replay_history(matches))

    if os.path.exists(LOG_PATH):
        log = pd.read_csv(LOG_PATH).fillna("")
    else:
        log = pd.DataFrame(columns=COLUMNS)

    # Log predictions for any fixture not already recorded (pre-match ratings).
    logged = set(zip(log["round"], log["home"], log["away"]))
    new_rows = []
    for rnd, home, away in FIXTURES:
        if (rnd, home, away) in logged:
            continue
        adv, prob, exp = predict(home, away, ratings, params)
        new_rows.append({"round": rnd, "home": home, "away": away,
                         "pred_advance": adv, "pred_prob": prob, "exp_score": exp,
                         "actual_advance": "", "actual_score": "",
                         "decided_by": "", "hit": ""})
    if new_rows:
        additions = pd.DataFrame(new_rows)
        log = additions if log.empty else pd.concat([log, additions],
                                                     ignore_index=True)

    # Score any tie whose actual advancer has been filled in.
    def score(row):
        actual = str(row["actual_advance"]).strip()
        return int(actual == row["pred_advance"]) if actual else ""

    log["hit"] = log.apply(score, axis=1)
    log = log[COLUMNS]
    log.to_csv(LOG_PATH, index=False)

    show = ["round", "home", "away", "pred_advance", "pred_prob",
            "exp_score", "actual_advance", "hit"]
    print("Knockout predictions log:")
    print(log[show].to_string(index=False))

    done = log[log["actual_advance"].astype(str).str.strip() != ""]
    if len(done):
        hits = done["hit"].astype(int)
        p = done["pred_prob"].astype(float) / 100.0
        brier = ((p - hits) ** 2).mean()
        print(f"\nRolling accuracy: {hits.sum()}/{len(done)} = "
              f"{100 * hits.mean():.0f}%   Brier score: {brier:.3f} "
              f"(lower is better; 0.25 = coin-flip)")
    else:
        print("\nPredictions locked in. Awaiting results - send them as they land.")


if __name__ == "__main__":
    main()
