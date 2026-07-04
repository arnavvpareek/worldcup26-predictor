"""
Gather recent international xG for the 2026 World Cup teams from StatsBomb's
free open data (github.com/statsbomb/open-data).

For each team we take its most recent major tournament with event data:
  * UEFA teams   -> Euro 2024
  * CONMEBOL/CONCACAF guests -> Copa America 2024
  * CAF teams    -> Africa Cup of Nations 2023
Shot-level xG is aggregated into expected goals for / against per game, plus
the actual goals, so a later step can measure how far each team's *results*
ran ahead of or behind its *chances*. (Norway isn't in any of these, so it
has no xG and simply receives no adjustment.)
"""

import json
import os
import urllib.request

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
CACHE = ".xgcache"
OUT = "data/raw/team_xg.csv"

# tournament label -> (competition_id, season_id, teams we want from it)
TOURNAMENTS = {
    "Euro 2024": (55, 282,
                  ["France", "Spain", "Portugal", "England", "Belgium",
                   "Switzerland"]),
    "Copa America 2024": (223, 282,
                          ["Argentina", "Brazil", "Colombia", "Paraguay",
                           "United States", "Canada", "Mexico"]),
    "AFCON 2023": (1267, 107, ["Morocco", "Egypt"]),
}


def get_json(url: str, cache_name: str):
    """Fetch JSON with a simple on-disk cache so re-runs are instant."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, cache_name)
    if not os.path.exists(path):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=45) as r:
            data = r.read()
        with open(path, "wb") as f:
            f.write(data)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def match_xg(events: list) -> tuple[dict, dict]:
    """Sum shot xG and count goals for each team in one match.

    Period 5 is the penalty shootout — those spot-kicks are a tiebreak, not
    chances created in the game, so they're excluded (otherwise a team that
    went to a shootout gets ~4 phantom xG of penalties).
    """
    xg, goals = {}, {}
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        if e.get("period") == 5:  # penalty shootout, not open play
            continue
        team = e["team"]["name"]
        shot = e.get("shot", {})
        xg[team] = xg.get(team, 0.0) + shot.get("statsbomb_xg", 0.0)
        if shot.get("outcome", {}).get("name") == "Goal":
            goals[team] = goals.get(team, 0) + 1
    return xg, goals


def main() -> None:
    agg: dict[str, dict] = {}

    for label, (comp, season, wanted) in TOURNAMENTS.items():
        matches = get_json(f"{BASE}/matches/{comp}/{season}.json",
                            f"matches_{comp}_{season}.json")
        wanted = set(wanted)
        # only download events for matches involving a team we care about
        rel = [m for m in matches
               if m["home_team"]["home_team_name"] in wanted
               or m["away_team"]["away_team_name"] in wanted]
        print(f"{label}: {len(rel)} relevant matches")

        for m in rel:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            try:
                events = get_json(f"{BASE}/events/{m['match_id']}.json",
                                  f"events_{m['match_id']}.json")
            except Exception as exc:  # noqa: BLE001
                print(f"  skip {home} v {away}: {exc}")
                continue
            xg, goals = match_xg(events)

            for team, opp in ((home, away), (away, home)):
                if team not in wanted:
                    continue
                a = agg.setdefault(team, {"tournament": label, "games": 0,
                                          "xg_for": 0.0, "xg_against": 0.0,
                                          "goals_for": 0, "goals_against": 0})
                a["games"] += 1
                a["xg_for"] += xg.get(team, 0.0)
                a["xg_against"] += xg.get(opp, 0.0)
                a["goals_for"] += goals.get(team, 0)
                a["goals_against"] += goals.get(opp, 0)

    rows = []
    for team, a in agg.items():
        g = a["games"]
        rows.append({
            "team": team, "tournament": a["tournament"], "games": g,
            "xg_for_pg": round(a["xg_for"] / g, 3),
            "xg_against_pg": round(a["xg_against"] / g, 3),
            "goals_for_pg": round(a["goals_for"] / g, 3),
            "goals_against_pg": round(a["goals_against"] / g, 3),
        })
    out = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"\nSaved {len(out)} teams -> {OUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
