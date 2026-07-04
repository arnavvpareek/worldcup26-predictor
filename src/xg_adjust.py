"""
Phase 6 - xG rating adjustment.

Elo is built on scorelines, which are noisy: a team that wins on a lucky
deflection is rated as if it dominated. Using real StatsBomb xG from each
team's most recent major tournament, we nudge ratings toward what the team
*deserved*:

  luck = actual goal-diff/game - xG goal-diff/game
  * luck > 0  -> results ran ahead of chances (lucky)   -> shave rating
  * luck < 0  -> results lagged chances (unlucky)        -> bump rating

The nudge is converted to Elo points via the calibrated "goals per Elo",
shrunk (one tournament is a small sample) and capped, so xG refines Elo rather
than overriding it. Teams with no xG data (e.g. Norway) are left unchanged.
"""

import sys

import pandas as pd

sys.path.insert(0, "src")
from elo import load_matches, run_elo  # noqa: E402
from simulate import calibrate, replay_history  # noqa: E402

XG_PATH = "data/raw/team_xg.csv"
TRUST = 0.20        # fraction of the xG residual we believe (one tournament)
GAMES_PRIOR = 4.0   # sample-size shrinkage: a team with this many games is
                    # trusted ~half; fewer games -> smaller nudge
CAP = 30.0          # rare safety limit on rating points moved


def compute_adjustments(params: dict, xg_path: str = XG_PATH):
    """Return (adjustment dict, detail DataFrame) from the xG table.

    adjustment = -luck * (Elo per goal) * TRUST * confidence, capped, where
    confidence = games / (games + GAMES_PRIOR) down-weights small samples.
    """
    df = pd.read_csv(xg_path)
    elo_per_goal = 1.0 / params["margin_slope"]  # ~174 from calibration

    adj, rows = {}, []
    for r in df.itertuples(index=False):
        xgd = r.xg_for_pg - r.xg_against_pg          # deserved goal diff/game
        gd = r.goals_for_pg - r.goals_against_pg     # actual goal diff/game
        luck = gd - xgd                              # >0 => overperformed xG
        confidence = r.games / (r.games + GAMES_PRIOR)
        raw = -luck * elo_per_goal * TRUST * confidence  # lucky => shave
        a = max(-CAP, min(CAP, raw))
        adj[r.team] = a
        rows.append((r.team, r.tournament, r.games, round(xgd, 2),
                     round(gd, 2), round(luck, 2), round(a, 1)))

    detail = pd.DataFrame(rows, columns=["team", "tournament", "games",
                                         "xgd_pg", "gd_pg", "luck", "adj"])
    return adj, detail.sort_values("adj").reset_index(drop=True)


def adjusted_ratings(ratings: dict, params: dict) -> dict:
    """Elo ratings with the capped xG nudge applied."""
    adj, _ = compute_adjustments(params)
    out = dict(ratings)
    for team, a in adj.items():
        if team in out:
            out[team] += a
    return out


def main() -> None:
    from bracket import run_monte_carlo  # local import to avoid cycle

    matches = load_matches()
    base = run_elo(matches).set_index("team")["rating"].to_dict()
    params = calibrate(replay_history(matches))

    adj, detail = compute_adjustments(params)
    print("xG adjustment per team (most-recent major tournament):")
    print(detail.to_string(index=False))
    print(f"\n(trust={TRUST}, games_prior={GAMES_PRIOR:.0f}, cap=+/-{CAP:.0f} "
          f"Elo; +adj = xG says underrated)\n")

    adj_ratings = {t: base[t] + adj.get(t, 0.0) for t in base}

    before = run_monte_carlo(base, params)
    after = run_monte_carlo(adj_ratings, params)

    out_path = "data/processed/tournament_probabilities_xg.csv"
    after.round(1).to_csv(out_path, index=False)

    cmp = pd.DataFrame({
        "champion_base": before.set_index("team")["champion"],
        "champion_xg": after.set_index("team")["champion"],
    })
    cmp["delta"] = (cmp["champion_xg"] - cmp["champion_base"]).round(1)
    cmp = cmp.sort_values("champion_xg", ascending=False)
    print("Title odds before vs after xG adjustment (%):")
    print(cmp.to_string())
    print(f"\nSaved xG-adjusted table -> {out_path}")


if __name__ == "__main__":
    main()
