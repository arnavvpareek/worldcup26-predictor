# World Cup 2026 Predictor

A from-scratch statistical model that predicts the knockout stage of the 2026
FIFA World Cup. No black-box classifier — every number is explainable:

**Elo ratings** (team strength from history) → **Poisson goal model** (scores
per match) → **Monte Carlo** (simulate the bracket thousands of times).

## How it works at a glance

The model is a pipeline: each stage's output feeds the next. We turn decades of
results into a strength number per team, use those numbers to simulate any
single match, then simulate the whole knockout draw thousands of times to see
how often each team lifts the trophy.

<p align="center"><img src="docs/pipeline.svg" alt="Predictor pipeline: historical results to Elo ratings to match simulator to shootout model to Monte Carlo bracket to champion probabilities" width="620"></p>

The engine that starts it all, Elo, rates every team with a single number.
Before a match it turns the **gap** between two ratings into a win probability
using an S-shaped (logistic) curve — a bigger lead means a higher, but never
certain, chance of winning:

<p align="center"><img src="docs/elo_curve.svg" alt="Logistic curve mapping Elo rating difference to win probability" width="680"></p>

## Status

| Phase | What it does | State |
|---|---|---|
| 1 | Elo rating engine built from historical results | ✅ done |
| 2 | Match simulator (Elo → win prob → Poisson goals) | ⏳ next |
| 3 | Penalty-shootout calibration from `shootouts.csv` | — |
| 4 | Monte Carlo bracket simulation | — |
| 5 | Output, methodology write-up, predicted-vs-actual tracking | — |

## Phase 1 — the Elo engine

`src/elo.py` replays every international match since 2000 (25,425 games) in
date order and maintains a running strength rating for all 321 national teams.

For each match:

1. **Predict** the home team's expected score from the rating gap, via the
   logistic formula

$$E_{\text{home}} = \frac{1}{1 + 10^{-(R_{\text{home}} - R_{\text{away}})/400}}$$

   A 400-point gap ≈ a 10-to-1 favorite. Home advantage (+65) is added to the
   expectation, but dropped on neutral ground (all World Cup knockouts).
2. **Observe** the actual result (win 1.0 / draw 0.5 / loss 0.0).
3. **Update** `R_new = R_old + K · G · (actual − expected)`.

`K` (learning rate) is weighted by match importance so a World Cup knockout
moves ratings more than a friendly:

| Tier | K |
|---|---|
| World Cup finals | 60 |
| Continental finals (Euro, Copa, AFCON, Asian Cup, Gold Cup, Confed) | 50 |
| Qualifiers / Nations League | 40 |
| Minor tournaments | 30 |
| Friendly | 20 |

`G` is a goal-difference multiplier: a 5-0 win is stronger evidence than a 1-0,
so bigger margins move ratings more (sub-linearly).

**Known model property:** penalty-shootout results are stored in `results.csv`
as draws (only the 120-minute score is recorded), so a knockout won on
penalties earns no "win" bump. `shootouts.csv` exists to correct for this and
is used in Phase 3.

### Run it

```bash
pip install -r requirements.txt
python src/elo.py
```

Outputs the top-25 ranking and writes the full table to
`data/processed/elo_ratings.csv`.

## Data

Kaggle: *International football results from 1872 to 2026* (martj42).
`data/raw/results.csv` (match results) and `data/raw/shootouts.csv`
(penalty-shootout winners).
