"""
Phase 1 — Elo rating engine (built from scratch).

Reads historical international match results, replays them in chronological
order, and maintains a running Elo rating for every national team. The output
is each team's *current* strength as of the most recent match in the data,
which later phases turn into match win-probabilities and a simulated bracket.

Core Elo loop
-------------
For each match, in date order:
    1. Predict     -> expected score from the rating gap (logistic curve)
    2. Observe     -> actual score (win = 1.0, draw = 0.5, loss = 0.0)
    3. Update      -> R_new = R_old + K * G * (actual - expected)

`K` is the learning rate (weighted by match importance) and `G` is a
goal-difference multiplier (bigger wins move ratings more). Whatever one team
gains the other loses, so the system is zero-sum and self-calibrating.
"""

import os

import pandas as pd


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_RATING = 1500.0   # every team's starting strength before it plays a game
HOME_ADV = 65.0        # rating points a team is effectively worth at home
SCALE = 400.0          # Elo scale: a 400-point gap == a 10-to-1 favorite
START_YEAR = 2000      # learn only from modern football (brief's instruction)

# K-factor by match importance. A World Cup knockout should shift ratings far
# more than a friendly, so more important matches carry more information.
K_WORLD_CUP = 60.0
K_CONTINENTAL = 50.0
K_QUALIFIER = 40.0
K_MINOR = 30.0
K_FRIENDLY = 20.0


def classify_k(tournament: str) -> float:
    """Map a tournament name to its K-factor tier.

    There are 100+ distinct tournament strings in the data, so we classify by
    keyword rather than hand-listing every one. Order matters: we check the
    most important tiers first and fall through to the least.
    """
    t = tournament.lower()

    # Friendlies carry the least information about true strength.
    if t == "friendly":
        return K_FRIENDLY

    # Qualifiers and Nations League: competitive but not the main event.
    # Checked before the continental block so "FIFA World Cup qualification"
    # doesn't get mistaken for the World Cup finals itself.
    if "qualif" in t or "nations league" in t:
        return K_QUALIFIER

    # The World Cup finals — the single most important tournament.
    if "fifa world cup" in t:
        return K_WORLD_CUP

    # Major continental finals + the Confederations Cup.
    continental = (
        "uefa euro", "copa am", "african cup of nations", "afc asian cup",
        "gold cup", "confederations cup",
    )
    if any(key in t for key in continental):
        return K_CONTINENTAL

    # Everything else: regional cups, minor invitationals, etc.
    return K_MINOR


def expected_score(rating_home: float, rating_away: float, neutral: bool) -> float:
    """Probability-like expectation that the home team wins (0..1).

    Logistic function of the rating gap. Home advantage is folded into the
    *expectation only* (not stored in the rating) and dropped on neutral
    ground — which is where every World Cup knockout is played.
    """
    home_effective = rating_home + (0.0 if neutral else HOME_ADV)
    gap = home_effective - rating_away
    return 1.0 / (1.0 + 10 ** (-gap / SCALE))


def goal_diff_multiplier(margin: int) -> float:
    """Scale the update by winning margin (World Football Elo convention).

    A 5-0 win is stronger evidence of dominance than a 1-0 win, so it should
    move ratings more. Margins of 0/1 -> 1.0, and the multiplier grows,
    but sub-linearly, for blowouts (a 6th goal matters less than the 2nd).
    """
    margin = abs(margin)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11 + margin) / 8.0


def actual_score(home_score: int, away_score: int) -> float:
    """Result from the home team's perspective: win=1.0, draw=0.5, loss=0.0."""
    if home_score > away_score:
        return 1.0
    if home_score < away_score:
        return 0.0
    return 0.5


def run_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Replay every match in date order and return final ratings per team.

    A plain Python loop is used deliberately: each update depends on the
    ratings produced by all prior matches, so the computation is inherently
    sequential and cannot be vectorized.
    """
    ratings: dict[str, float] = {}       # team -> current Elo
    matches_played: dict[str, int] = {}  # team -> games processed
    last_played: dict[str, str] = {}     # team -> date of most recent game

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team

        # A team's first appearance starts it at the baseline.
        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        # 1. Predict (home perspective). Away expectation is the complement,
        #    because in a two-outcome model the probabilities sum to 1.
        exp_home = expected_score(r_home, r_away, row.neutral)
        exp_away = 1.0 - exp_home

        # 2. Observe.
        act_home = actual_score(row.home_score, row.away_score)
        act_away = 1.0 - act_home

        # 3. Update. Both teams share the same K (match importance) and G
        #    (margin), and their surprises are equal and opposite.
        k = classify_k(row.tournament)
        g = goal_diff_multiplier(row.home_score - row.away_score)
        ratings[home] = r_home + k * g * (act_home - exp_home)
        ratings[away] = r_away + k * g * (act_away - exp_away)

        for team in (home, away):
            matches_played[team] = matches_played.get(team, 0) + 1
            last_played[team] = row.date

    out = pd.DataFrame(
        {
            "team": list(ratings.keys()),
            "rating": [round(ratings[t], 1) for t in ratings],
            "matches": [matches_played[t] for t in ratings],
            "last_match": [last_played[t] for t in ratings],
        }
    )
    return out.sort_values("rating", ascending=False).reset_index(drop=True)


def replay_history(df: pd.DataFrame) -> pd.DataFrame:
    """Replay every match and record each team's rating *before* it was played.

    Same loop as `run_elo`, but instead of only the final table it returns one
    row per match with the pre-match ratings and the effective rating gap
    (home advantage folded in, dropped on neutral ground). Phase 2 uses this
    to calibrate goals-vs-Elo and to backtest predictions on pre-match state.
    """
    ratings: dict[str, float] = {}
    records = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        r_home = ratings.get(home, BASE_RATING)
        r_away = ratings.get(away, BASE_RATING)

        eff_gap = r_home + (0.0 if row.neutral else HOME_ADV) - r_away
        records.append(
            {
                "date": row.date,
                "home_team": home,
                "away_team": away,
                "home_score": row.home_score,
                "away_score": row.away_score,
                "tournament": row.tournament,
                "neutral": row.neutral,
                "home_elo_pre": r_home,
                "away_elo_pre": r_away,
                "eff_gap": eff_gap,
            }
        )

        exp_home = expected_score(r_home, r_away, row.neutral)
        act_home = actual_score(row.home_score, row.away_score)
        k = classify_k(row.tournament)
        g = goal_diff_multiplier(row.home_score - row.away_score)
        ratings[home] = r_home + k * g * (act_home - exp_home)
        ratings[away] = r_away + k * g * ((1 - act_home) - (1 - exp_home))

    return pd.DataFrame(records)


def load_matches(
    path: str = "data/raw/results.csv",
    start_year: int = START_YEAR,
    manual_path: str = "data/raw/results_2026_manual.csv",
) -> pd.DataFrame:
    """Load results.csv, keep completed matches from `start_year` on, sort.

    `results.csv` ships the full 2026 fixture list with blank scores for
    unplayed games (dropped below). As real results land after the data's
    cutoff, we append them via `manual_path` so every phase runs on ratings
    that stay current — this is also the hook Phase 5 uses to log results.
    """
    df = pd.read_csv(path)

    # Append any manually-entered post-cutoff results (e.g. live WC games).
    if manual_path and os.path.exists(manual_path):
        df = pd.concat([df, pd.read_csv(manual_path)], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])

    # Drop any rows without a recorded score (future/blank fixtures).
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalize the neutral flag ("TRUE"/"FALSE" strings) into a real bool.
    df["neutral"] = df["neutral"].astype(str).str.strip().str.upper() == "TRUE"

    df = df[df["date"].dt.year >= start_year]

    # Chronological order is essential — the loop depends on prior state.
    df = df.sort_values("date").reset_index(drop=True)
    # Store date as a plain string for clean output.
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main() -> None:
    matches = load_matches("data/raw/results.csv")
    print(f"Matches replayed (since {START_YEAR}): {len(matches):,}")

    ratings = run_elo(matches)
    print(f"Teams rated: {len(ratings):,}")

    out_path = "data/processed/elo_ratings.csv"
    ratings.to_csv(out_path, index=False)
    print(f"Saved full ratings table -> {out_path}\n")

    print("Top 25 national teams by current Elo:")
    top = ratings.head(25).copy()
    top.index = range(1, len(top) + 1)
    print(top.to_string())


if __name__ == "__main__":
    main()
