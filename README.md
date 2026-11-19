# GTA VI Trophy Rarity Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nonicon/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nonicon/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase one week after launch (26 November 2026, 00:00 UTC) — from the house's reads of the trophy page before the case's checkpoint. Entry for the make.fun bounty [`gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) ($250 total, top 20 paid; score is mean pinball loss on the logit scale, lower is better).

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic) reads one case from `/input/<id>.in` and writes `/output/<id>.out`:

1. **Decomposition.** A trophy's earn rate is earners / players tracked, and the two move for different reasons: earners grow as players progress through the game, players tracked grows as more players link Exophase. The launch-night refresh proved it: when the tracked base jumped 638 → 3,868, every rate but the 100% one fell, because the new tracked players had not earned those trophies yet. So the forecast models the two series separately and takes their ratio, instead of extrapolating the rate itself.
2. **Growth laws.** Both series are fit, per trophy (earners) and once (players), as log-linear in log time since the first unlock (New Zealand, 13 hours before the release date's UTC midnight). Log time already decelerates growth, so a trophy rising fast now is not extrapolated to 90% by the target. Fits use only the distinct reads (Exophase refreshes infrequently, so most reads repeat the last page) and weight recent reads more (24-hour half-life).
3. **Damping and caps.** Earners never fall (floored at the last read's count); the tracked base is capped at 100× the last read; the predicted log-rate move from the last read is capped at ±1.5; each trophy's median is capped by a tier/description ceiling (platinum 1%, gold 25%, silver 35%, early story bronze 70% falling with chapter number, other bronze 40%). The 100% trophy stays 100%. When the history is flat (fewer than 3 distinct reads), the median is shifted down 0.5 logit for the dilution a refresh is about to bring.
4. **Spread.** The predictive spread on the logit scale combines a random-walk term (0.5 per √day, the house baseline's), the fit uncertainty of the two log-linear fits, and a refresh term (1.0 while the history is flat, 0.5 once it has shown several refreshes): Exophase refreshes move every rate at once, so the band is wide enough to cover a refresh the case has not seen yet.

## Validation

Scored locally on the two practice cases against the rates the live Exophase page showed after the launch-night refresh (the practice answers; the house baseline scores 1.508 here, matching the rules' "about 1.51"):

| | loss | typical miss | 90% coverage |
|---|---|---|---|
| This forecaster | **0.600** | 0.335 pts | 1.00 |
| House baseline | 1.508 | 0.677 pts | 0.00 |

On a synthetic final-like case (2.5-day history, 4.5-day horizon): 0.295 vs the house baseline's 0.304. Full numbers in `analysis/score-practice.txt`. The practice straddles one refresh over 90 minutes and is little guide to the final's week; the model is built for the final's dynamics (dilution as the tracked base grows, then progression as players advance).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=nj0kv9v2&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
