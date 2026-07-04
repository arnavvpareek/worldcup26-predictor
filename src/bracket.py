"""
Phase 4 & 5 - Monte Carlo bracket simulation and final output.

Plays the Round of 16 through to the final many thousands of times. Each match
is drawn from the Phase 2 goal model (Poisson scorelines), knockout ties are
broken by extra time then the Phase 3 shootout model, and winners advance up a
standard single-elimination tree. Counting the outcomes turns one bracket into
a probability for every team to reach each round and to win the cup.

Why simulate instead of just multiplying probabilities: a team's path depends
on who it meets next, which is itself random. Monte Carlo lets those branching
possibilities interact naturally — the law of large numbers does the rest.
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

# Round-of-16 entrants in bracket order, read off the official 2026 draw.
# Adjacent pairs are the R16 ties, and winners climb the tree, so the
# quarter-finals are (match1 vs match2), (match3 vs match4), ... The first
# eight teams are the left half of the draw, the last eight the right half —
# so France, Spain, Portugal and Belgium can only meet up to the semi-final.
R16_BRACKET = [
    # --- left half ---
    "Paraguay", "France",         # R16 match 1  \_ QF1
    "Canada", "Morocco",          # R16 match 2  /
    "Portugal", "Spain",          # R16 match 3  \_ QF2
    "United States", "Belgium",   # R16 match 4  /
    # --- right half ---
    "Brazil", "Norway",           # R16 match 5  \_ QF3
    "Mexico", "England",          # R16 match 6  /
    "Argentina", "Egypt",         # R16 match 7  \_ QF4
    "Switzerland", "Colombia",    # R16 match 8  /
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
