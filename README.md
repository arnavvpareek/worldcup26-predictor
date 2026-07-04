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

## Predicted results — who wins the 2026 World Cup?

Running the full knockout bracket **20,000 times** from the Round of 16 gives
every team a probability of reaching each round and lifting the trophy. Three
teams stand clear — Argentina, France and Spain — with France boosted by a
noticeably softer half of the draw.

<p align="center"><img src="docs/title_odds.svg" alt="Bar chart of each team's probability of winning the 2026 World Cup" width="680"></p>

| Team | Reach QF | Reach SF | Reach Final | **Win Cup** |
|---|---:|---:|---:|---:|
| Argentina | 92% | 50% | 34% | **24%** |
| France | 83% | 58% | 45% | **22%** |
| Spain | 67% | 38% | 26% | **17%** |
| Brazil | 72% | 45% | 19% | **10%** |
| England | 52% | 21% | 14% | **6%** |
| Colombia | 59% | 39% | 15% | **5%** |
| Morocco | 68% | 34% | 11% | **5%** |
| Mexico | 48% | 17% | 10% | **3%** |
| Portugal | 33% | 12% | 6% | **3%** |
| *others* | | | | **<2% each** |

The single most-likely bracket (favourite at every tie) ends **Spain champion**,
beating France in the final. Note this differs from the title-odds table: the
*modal path* and the team with the *highest overall win probability* need not
match, because Argentina and France reach the final via more (easier) routes
than Spain, who must pass Argentina and Brazil in one half.

Full numbers: [`data/processed/tournament_probabilities.csv`](data/processed/tournament_probabilities.csv).

## Status

| Phase | What it does | State |
|---|---|---|
| 1 | Elo rating engine built from historical results | ✅ done |
| 2 | Match simulator (Elo → Poisson goals → outcomes) | ✅ done |
| 3 | Penalty-shootout calibration from `shootouts.csv` | ✅ done |
| 4 | Monte Carlo bracket simulation | ✅ done |
| 5 | Output tables + predicted-vs-actual tracking | ✅ done |

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

## Phase 2 — the match simulator

`src/simulate.py` turns a rating gap into a scoreline. The goals-vs-Elo
relationship is **calibrated from history, not assumed**: binning 25k matches
shows ~1 goal of expected margin per 174 Elo points, and total goals rise
slightly with the mismatch. Each team's expected goals (λ) feed a **Poisson
distribution**; crossing the two teams' distributions gives the probability of
every scoreline, and hence win / draw / loss.

Validated on pre-match ratings only: predicted draw rate 21% vs 23% actual,
average goals 2.76 vs 2.76, and the Elo favourite wins **76.6%** of decided
matches across 12,000+ games since 2010 (13/13 on the 2026 R32 knockouts).

## Phase 3 — shootout calibration

`shootouts.csv` says the favourite wins a shootout only **~54%** of the time, so
penalties are close to a coin flip. `shootout_prob` reuses the logistic curve
with a far wider scale (1250 vs 400) — a level tie is 50/50, a big favourite
gets only a mild edge — replacing the earlier flat assumption for knockout ties.

## Phase 4 & 5 — Monte Carlo bracket and output

`src/bracket.py` plays the R16 → final **20,000 times**. Every tie is drawn from
the Phase 2 goal model, level games go to extra time then the Phase 3 shootout,
and winners advance up the tree. Counting the outcomes yields each team's
probability of reaching every round and winning the cup (the results above),
saved to `data/processed/tournament_probabilities.csv`.

## Run it

```bash
pip install -r requirements.txt
python src/elo.py        # Phase 1 — ratings -> data/processed/elo_ratings.csv
python src/simulate.py   # Phase 2/3 — calibration, backtest, R16 predictions
python src/bracket.py    # Phase 4/5 — full tournament odds + predicted bracket
```

## Data

Kaggle: *International football results from 1872 to 2026* (martj42).
`data/raw/results.csv` (match results) and `data/raw/shootouts.csv`
(penalty-shootout winners).
