"""
Phase 4 & 5 - Monte Carlo bracket simulation and final output.

Plays the Round of 16 through to the final many thousands of times. Each match
is drawn from the Phase 2 goal model (Poisson scorelines), knockout ties are
broken by extra time then the Phase 3 shootout model, and winners advance up a
standard single-elimination tree. Counting the outcomes turns one bracket into
a probability for every team to reach each round and to win the cup.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from elo import load_matches, run_elo  # noqa: E402
from simulate import (  # noqa: E402
    advance_prob,
    calibrate,
    expected_goals,
    replay_history,
    shootout_prob,
    simulate_match,
)

N_SIMS = 20000
SEED = 42

# Remaining entrants in bracket order, updated as each real round completes.
# Adjacent pairs are the current round's ties; winners climb the tree.
# Quarter-finals after the real R16 results (left half first):
R16_BRACKET = [
    # --- left half ---
    "France", "Morocco",          # QF1 \_ SF1
    "Spain", "Belgium",           # QF2 /
    # --- right half ---
    "Norway", "England",          # QF3 \_ SF2
    "Argentina", "Switzerland",   # QF4 /
]

# Winners of a round of this size have "reached" the next stage.
STAGE_BY_SIZE = {8: "reach_QF", 4: "reach_SF", 2: "reach_final", 1: "champion"}


def play_match(a: str, b: str, ratings: dict, params: dict, rng) -> str:
    """Simulate one knockout tie and return the winning team name."""
    ea, eb = ratings[a], ratings[b]
    res = simulate_match(
        ea, eb, neutral=True, params=params, rng=rng,
        knockout=True, pen_home=shootout_prob(ea, eb),
    )
    return a if res["winner"] == "home" else b


def simulate_once(ratings: dict, params: dict, rng, counts: dict) -> str:
    """Play one whole bracket; tally each round reached. Return the champion."""
    teams = list(R16_BRACKET)
    while len(teams) > 1:
        winners = [
            play_match(teams[i], teams[i + 1], ratings, params, rng)
            for i in range(0, len(teams), 2)
        ]
        teams = winners
        stage = STAGE_BY_SIZE[len(teams)]
        for w in winners:
            counts[w][stage] += 1
    return teams[0]


def run_monte_carlo(ratings: dict, params: dict) -> pd.DataFrame:
    """Run N_SIMS full-tournament simulations and return probabilities."""
    rng = np.random.default_rng(SEED)
    stages = ["reach_QF", "reach_SF", "reach_final", "champion"]
    counts = {t: {s: 0 for s in stages} for t in R16_BRACKET}

    for _ in range(N_SIMS):
        simulate_once(ratings, params, rng, counts)

    rows = []
    for team in R16_BRACKET:
        row = {"team": team, "elo": round(ratings[team])}
        for s in stages:
            row[s] = 100 * counts[team][s] / N_SIMS
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("champion", ascending=False)
    return out.reset_index(drop=True)


def chalk_bracket(ratings: dict, params: dict) -> None:
    """Print the single most-likely path: pick the favorite at every tie."""
    round_names = {16: "Round of 16", 8: "Quarter-finals",
                   4: "Semi-finals", 2: "Final"}
    teams = list(R16_BRACKET)
    while len(teams) > 1:
        print(f"  {round_names[len(teams)]}:")
        winners = []
        for i in range(0, len(teams), 2):
            a, b = teams[i], teams[i + 1]
            la, lb = expected_goals(ratings[a], ratings[b], True, params)
            p = advance_prob(la, lb, pen_home=shootout_prob(ratings[a], ratings[b]))
            fav, pf = (a, p) if p >= 0.5 else (b, 1 - p)
            print(f"    {a} vs {b:16s} -> {fav} ({pf:.0%})")
            winners.append(fav)
        teams = winners
    print(f"  Predicted champion: {teams[0]}")


def main() -> None:
    matches = load_matches()
    ratings = run_elo(matches).set_index("team")["rating"].to_dict()
    params = calibrate(replay_history(matches))

    missing = [t for t in R16_BRACKET if t not in ratings]
    if missing:
        raise SystemExit(f"Teams missing from ratings: {missing}")

    print(f"Running {N_SIMS:,} tournament simulations...\n")
    table = run_monte_carlo(ratings, params)

    out_path = "data/processed/tournament_probabilities.csv"
    table.round(1).to_csv(out_path, index=False)

    print("Title odds and run probabilities (%):")
    print("  %-15s %5s %7s %7s %8s %8s"
          % ("team", "elo", "QF", "SF", "final", "CHAMP"))
    for r in table.itertuples(index=False):
        print("  %-15s %5d %6.1f %6.1f %7.1f %7.1f"
              % (r.team, r.elo, r.reach_QF, r.reach_SF, r.reach_final, r.champion))
    print(f"\nSaved -> {out_path}\n")

    print("Most-likely bracket (favorite at every tie):")
    chalk_bracket(ratings, params)


if __name__ == "__main__":
    main()
