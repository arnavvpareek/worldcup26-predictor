"""
Phase 2 — match simulator.

Turns Elo ratings into concrete match outcomes in two layers:

    Elo gap  --calibrated regression-->  each team's goal rate (lambda)
    lambda   --Poisson distribution --->  a full scoreline probability grid

From the grid we read off P(home win) / P(draw) / P(away win) analytically,
and for knockouts we resolve draws with extra time then penalties. The same
model can be sampled (`simulate_match`) to drive the Phase 4 Monte Carlo.

Nothing about scoring is assumed: the goals-vs-Elo relationship is fit from
the historical data itself (see `calibrate`).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from elo import (  # noqa: E402
    HOME_ADV,
    SCALE,
    expected_score,
    load_matches,
    replay_history,
    run_elo,
)

# Extra time is 30 minutes vs 90, so goal rates scale by ~1/3.
EXTRA_TIME_FRACTION = 30.0 / 90.0
# Max goals per team when building the analytic scoreline grid (>=10 covers
# virtually all realistic matches; the Poisson tail beyond it is negligible).
MAX_GOALS = 10


# --------------------------------------------------------------------------- #
# Calibration: fit goals as a function of the Elo gap, from history
# --------------------------------------------------------------------------- #
def calibrate(history: pd.DataFrame) -> dict:
    """Fit two linear relationships from historical (pre-match) data.

    * goal margin (home - away) vs the effective Elo gap
    * total goals (home + away) vs the *absolute* gap (mismatches are higher-
      scoring because the favorite piles on)

    Returns the four regression coefficients. We deliberately model the sum
    and the difference of goals rather than each team directly: it keeps the
    fit to two simple, inspectable regressions and guarantees the split is
    internally consistent.
    """
    gap = history["eff_gap"].to_numpy()
    margin = (history["home_score"] - history["away_score"]).to_numpy()
    total = (history["home_score"] + history["away_score"]).to_numpy()

    margin_slope, margin_intercept = np.polyfit(gap, margin, 1)
    total_slope, total_intercept = np.polyfit(np.abs(gap), total, 1)

    return {
        "margin_slope": float(margin_slope),
        "margin_intercept": float(margin_intercept),
        "total_slope": float(total_slope),
        "total_intercept": float(total_intercept),
    }


def expected_goals(
    elo_home: float, elo_away: float, neutral: bool, params: dict
) -> tuple[float, float]:
    """Map two ratings to each team's expected goals (Poisson lambda)."""
    eff_gap = elo_home + (0.0 if neutral else HOME_ADV) - elo_away
    margin = params["margin_slope"] * eff_gap + params["margin_intercept"]
    total = params["total_slope"] * abs(eff_gap) + params["total_intercept"]

    lam_home = (total + margin) / 2.0
    lam_away = (total - margin) / 2.0
    # Goal rates must be positive; clamp the rare extreme-mismatch case.
    return max(0.05, lam_home), max(0.05, lam_away)


# --------------------------------------------------------------------------- #
# Analytic outcome probabilities (no Monte Carlo noise)
# --------------------------------------------------------------------------- #
def _poisson_pmf(lam: float, max_k: int = MAX_GOALS) -> np.ndarray:
    """Probability of 0..max_k goals for rate lam (last cell absorbs the tail)."""
    ks = np.arange(max_k + 1)
    pmf = np.exp(-lam) * lam**ks / np.array([math.factorial(k) for k in ks])
    pmf[-1] += 1.0 - pmf.sum()  # dump the remaining tail mass into the top cell
    return pmf


def outcome_probs(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    """P(home win), P(draw), P(away win) over 90 minutes, computed exactly.

    Cross the two independent Poissons into a scoreline grid and sum the cells
    below / on / above the diagonal.
    """
    ph = _poisson_pmf(lam_home)
    pa = _poisson_pmf(lam_away)
    grid = np.outer(ph, pa)  # grid[i, j] = P(home i, away j)
    p_draw = np.trace(grid)
    p_home = np.tril(grid, -1).sum()  # home goals (row) > away goals (col)
    p_away = np.triu(grid, 1).sum()
    return p_home, p_draw, p_away


def advance_prob(
    lam_home: float, lam_away: float, pen_home: float = 0.5
) -> float:
    """Probability the HOME side advances from a knockout tie.

    Win in regulation, else survive extra time (goal rates scaled to 30 min),
    else win the shootout. `pen_home` is the home shootout win-prob — a 50/50
    placeholder here; Phase 3 replaces it with an Elo-calibrated value.
    """
    p_home, p_draw, _ = outcome_probs(lam_home, lam_away)

    et_home, et_draw, _ = outcome_probs(
        lam_home * EXTRA_TIME_FRACTION, lam_away * EXTRA_TIME_FRACTION
    )
    survive_draw = et_home + et_draw * pen_home
    return p_home + p_draw * survive_draw


def win_probability(elo_home: float, elo_away: float, neutral: bool) -> float:
    """Headline logistic-Elo expected score (blends win + half of draw).

    Kept as an independent cross-check on the Poisson model: this number
    should track (P_home_win + 0.5 * P_draw) from `outcome_probs`.
    """
    return expected_score(elo_home, elo_away, neutral)


# --------------------------------------------------------------------------- #
# Sampling (for the Phase 4 Monte Carlo bracket)
# --------------------------------------------------------------------------- #
def simulate_match(
    elo_home: float,
    elo_away: float,
    neutral: bool,
    params: dict,
    rng: np.random.Generator,
    knockout: bool = True,
    pen_home: float = 0.5,
) -> dict:
    """Simulate one match: draw goals from Poisson, resolve knockout ties."""
    lam_home, lam_away = expected_goals(elo_home, elo_away, neutral, params)
    gh = int(rng.poisson(lam_home))
    ga = int(rng.poisson(lam_away))

    result = {"home_goals": gh, "away_goals": ga, "decided_by": "regulation"}
    if gh != ga or not knockout:
        result["winner"] = "home" if gh > ga else ("away" if ga > gh else "draw")
        return result

    # Extra time.
    gh += int(rng.poisson(lam_home * EXTRA_TIME_FRACTION))
    ga += int(rng.poisson(lam_away * EXTRA_TIME_FRACTION))
    result.update(home_goals=gh, away_goals=ga, decided_by="extra_time")
    if gh != ga:
        result["winner"] = "home" if gh > ga else "away"
        return result

    # Penalties.
    result["decided_by"] = "penalties"
    result["winner"] = "home" if rng.random() < pen_home else "away"
    return result


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_poisson(history: pd.DataFrame, params: dict) -> None:
    """Sanity-check the Poisson layer: predicted vs actual draw rate & goals."""
    lam = history.apply(
        lambda r: expected_goals(
            r.home_elo_pre, r.away_elo_pre, r.neutral, params
        ),
        axis=1,
        result_type="expand",
    )
    pred_draw = [
        outcome_probs(lh, la)[1] for lh, la in zip(lam[0], lam[1])
    ]
    actual_draw = (history["home_score"] == history["away_score"]).mean()
    print("  Poisson draw rate  : predicted %.1f%%  vs actual %.1f%%"
          % (100 * np.mean(pred_draw), 100 * actual_draw))
    print("  Avg goals per match: predicted %.2f  vs actual %.2f"
          % ((lam[0] + lam[1]).mean(),
             (history["home_score"] + history["away_score"]).mean()))


def backtest(history: pd.DataFrame, params: dict, mask) -> pd.DataFrame:
    """For each selected match, pick the Elo favorite and check the result.

    Uses only pre-match ratings, so it is a genuine out-of-sample prediction.
    Draws are excluded from accuracy (there is no 'correct winner' to call).
    """
    sub = history[mask].copy()
    rows = []
    for r in sub.itertuples(index=False):
        p_home, p_draw, p_away = outcome_probs(
            *expected_goals(r.home_elo_pre, r.away_elo_pre, r.neutral, params)
        )
        pred = "home" if p_home >= p_away else "away"
        if r.home_score > r.away_score:
            actual = "home"
        elif r.away_score > r.home_score:
            actual = "away"
        else:
            actual = "draw"
        rows.append(
            {
                "match": f"{r.home_team} vs {r.away_team}",
                "score": f"{int(r.home_score)}-{int(r.away_score)}",
                "fav": r.home_team if pred == "home" else r.away_team,
                "p_fav": round(max(p_home, p_away), 3),
                "actual": actual,
                "hit": (actual != "draw" and pred == actual),
                "draw": actual == "draw",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    matches = load_matches()
    history = replay_history(matches)
    params = calibrate(history)

    print("Calibration (goals as a function of the Elo gap):")
    print("  goal margin = %.5f * gap  +  %.3f   (1 goal per %.0f Elo)"
          % (params["margin_slope"], params["margin_intercept"],
             1 / params["margin_slope"]))
    print("  total goals = %.5f * |gap| + %.3f"
          % (params["total_slope"], params["total_intercept"]))
    print()

    print("Poisson model validation (whole history):")
    validate_poisson(history, params)
    print()

    # --- Backtest 1: the brief's ask — 2026 World Cup knockout matches --- #
    history["dt"] = pd.to_datetime(history["date"])
    wc_ko = (
        (history["tournament"] == "FIFA World Cup")
        & (history["dt"] >= "2026-06-28")
    )
    bt = backtest(history, params, wc_ko)
    decided = bt[~bt["draw"]]
    print(f"Backtest - 2026 WC knockout matches ({len(bt)} played):")
    for r in bt.itertuples(index=False):
        flag = "DREW " if r.draw else ("HIT  " if r.hit else "miss ")
        print(f"  {flag} {r.match:34s} {r.score}  fav {r.fav} ({r.p_fav:.0%})")
    if len(decided):
        print("  => favorite called %d/%d decided games (%.0f%%)"
              % (decided["hit"].sum(), len(decided),
                 100 * decided["hit"].mean()))
    print()

    # --- Backtest 2: large historical sample for statistical weight --- #
    big = (history["dt"] >= "2010-01-01")
    bt_big = backtest(history, params, big)
    dec_big = bt_big[~bt_big["draw"]]
    print("Backtest - all matches since 2010 (statistical weight):")
    print("  decided games: %d, favorite correct %.1f%%, draw rate %.1f%%"
          % (len(dec_big), 100 * dec_big["hit"].mean(),
             100 * bt_big["draw"].mean()))
    print()

    # --- Deliverable: predict the confirmed Round-of-16 fixtures --- #
    ratings = run_elo(matches).set_index("team")["rating"].to_dict()
    r16 = [
        ("Canada", "Morocco"),
        ("Paraguay", "France"),
        ("Brazil", "Norway"),
        ("Mexico", "England"),
        ("Portugal", "Spain"),
        ("United States", "Belgium"),
    ]
    print("Round-of-16 predictions (neutral venue, knockout):")
    print("  %-24s %-8s %-8s  %s" % ("match", "goals", "advance", "favorite"))
    for home, away in r16:
        eh, ea = ratings[home], ratings[away]
        lh, la = expected_goals(eh, ea, neutral=True, params=params)
        adv_home = advance_prob(lh, la)
        fav, p = (home, adv_home) if adv_home >= 0.5 else (away, 1 - adv_home)
        print("  %-10s v %-10s  %.1f-%.1f   %4.0f%%    %s"
              % (home, away, lh, la, 100 * adv_home, f"{fav} {p:.0%}"))


if __name__ == "__main__":
    main()
