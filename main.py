"""GTA VI trophy rarity forecaster — make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for each of Grand Theft Auto VI's PlayStation 5 trophies, five
quantiles (5%, 25%, 50%, 75%, 95%) of its earn rate on Exophase at the
case's target, from the house's reads of the trophy page before the case's
checkpoint.

Method (stdlib only, deterministic, no network):

  * Decomposition. A trophy's earn rate is earners / players_tracked, and
    the two move for different reasons: earners grow as players progress
    through the game, players_tracked grows as more players link Exophase.
    The launch-night refresh proved it: when the tracked base jumped
    638 -> 3,868, every rate but the 100% one fell, because the new
    tracked players had not earned those trophies yet. So the forecast
    models the two series separately and takes their ratio, instead of
    extrapolating the rate itself (which would bake the dilution into a
    single trend and miss the turn when progression takes over).

  * Growth laws. Both series are fit, per trophy (earners) and once
    (players), as log-linear in log time since the first unlock
    (New Zealand, 13 hours before the release date's UTC midnight):
    ln E = a + b ln(1 + t). Log time already decelerates growth, so a
    trophy rising fast now is not extrapolated to 90% by the target.
    Fits use only the distinct reads (Exophase refreshes infrequently,
    so most reads repeat the last page) and weight recent reads more
    (24-hour half-life), so the stale launch-night page does not anchor
    the forecast.

  * Damping and caps. Earners never fall (floored at the last read's
    count); the tracked base is capped at 100x the last read; the
    predicted log-rate move from the last read is capped at +/-1.5;
    each trophy's median is capped by a tier/description ceiling
    (platinum 1%, gold 25%, silver 35%, early story bronze 70% falling
    with chapter number, other bronze 40%). The 100% trophy stays 100%.

  * Spread. The predictive spread on the logit scale combines a
    random-walk term (0.5 per sqrt day, the house baseline's), the fit
    uncertainty of the two log-linear fits, and a refresh term: Exophase
    refreshes move every rate at once (the launch-night refresh moved
    most rates about 1 logit unit), so the band is wide enough to cover
    a refresh the case has not seen yet. The refresh term is 1.0 while
    the history is flat and 0.5 once it has shown several refreshes.
    Quantiles are the median's logit plus the normal quantiles times
    the spread.

Validation (local, on the two practice cases, scored against the rates
the live Exophase page showed after the launch-night refresh — the
practice answers; the house baseline scores 1.508 here, matching the
rules' "about 1.51"): this forecaster 0.600 loss, typical miss 0.335
points, 90% coverage 1.00, against the house baseline's 1.508 / 0.677 /
0.00. On a synthetic final-like case (2.5-day history, 4.5-day horizon):
0.295 vs the house baseline's 0.304. The practice straddles one refresh
over 90 minutes and is little guide to the final's week; the model is
built for the final's dynamics (dilution then progression over 4.5 days).
Full numbers in analysis/score-practice.txt.

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, stdlib only).
It reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output). Two runs on the same
case write byte-identical output.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# module shipped beside this file would import (none is needed here).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Random-walk spread of a rate's move, logit units per square-root day.
RW_SIGMA = 0.5
# Extra spread for Exophase refresh shocks, logit units: one refresh can
# move every rate about a logit unit at once (launch night proved it).
# It is large when the history is flat (the refresh is still to come and
# unseen) and smaller once the history has shown several refreshes.
REFRESH_SIGMA_FLAT = 1.0
REFRESH_SIGMA_RICH = 0.5
# Largest log-rate move the median may make from the last read.
MAX_LOG_MOVE = 1.5
# When the history is flat (fewer than 3 distinct reads), the base has
# not refreshed yet in the window — but it will: Exophase's tracked base
# jumps when it refreshes, and a jump dilutes every rate. Shift the
# median down by this much (logit) so the forecast does not pretend the
# flat page will last. A rich history (the final case) never triggers it.
FLAT_HISTORY_READS = 3
FLAT_HISTORY_SHIFT = -0.5
# Largest growth of the tracked base the forecast may assume.
MAX_BASE_GROWTH = 100.0
# Recency half-life for the fits, in hours.
HALF_LIFE_H = 24.0

CHAPTER_RE = re.compile(r"chapter\s+(\d+)", re.IGNORECASE)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_ceiling(trophy: dict) -> float:
    """Upper bound on a trophy's earn rate at the target, as a fraction.

    From the trophy's tier and what it asks for: the platinum is 100%
    completion (nobody gets it in a week); gold is a major milestone;
    silver a mid-game goal; bronze story chapters fall with chapter
    number (early chapters most players reach, late ones few do).
    """
    desc = (trophy.get("description") or "") + " " + (trophy.get("name") or "")
    low = desc.lower()
    if "100% completion" in low or "100% complet" in low:
        return 0.02
    m = CHAPTER_RE.search(low)
    if m:
        n = int(m.group(1))
        return max(0.25, 0.75 - 0.05 * (n - 1))
    if "final mission" in low:
        return 0.45
    tier = (trophy.get("tier") or "").lower()
    if tier == "platinum":
        return 0.01
    if tier == "gold":
        return 0.25
    if tier == "silver":
        return 0.35
    return 0.40


def wls(xs, ys, ws):
    """Weighted least squares y = a + b x. Returns (a, b, resid_sd, sxx, xbar).

    With fewer than 2 points, or no spread in x, returns the weighted
    mean as the intercept and slope 0.
    """
    n = len(xs)
    sw = sum(ws)
    if n == 0 or sw <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / sw
    ybar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if n < 2 or sxx < 1e-12:
        resid = [y - ybar for y in ys]
        sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / sw) if n else 0.0
        return ybar, 0.0, sd, 0.0, xbar
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, n - 2)
    sd = math.sqrt(sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / n)) if n > 2 else 0.0
    return a, b, sd, sxx, xbar


def forecast_case(case: dict, name: str) -> dict:
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])

    # Time origin: the first unlock, New Zealand, 13 hours before the
    # release date's UTC midnight. Everything is measured in hours since
    # then, so log time decelerates growth the way adoption does.
    release = datetime.fromisoformat(case["game"]["release_date"] + "T00:00:00+00:00")
    origin = release - timedelta(hours=13)

    def hours_since_origin(dt: datetime) -> float:
        return max(0.5, (dt - origin).total_seconds() / 3600.0)

    t_target = hours_since_origin(target)

    # Distinct reads only: Exophase refreshes infrequently, so most reads
    # repeat the last page. A read counts as new when the tracked base or
    # any rate changed. The last read is always kept.
    distinct = []
    prev_key = None
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != prev_key:
            distinct.append(read)
            prev_key = key
    if distinct[-1] is not case["history"][-1]:
        distinct.append(case["history"][-1])

    last = distinct[-1]
    t_last = hours_since_origin(parse_time(last["at"]))
    n_last = last["players_tracked"] if last["players_tracked"] else 1

    # Recency weights: 24-hour half-life from the last read.
    def weight(read):
        age_h = max(0.0, t_last - hours_since_origin(parse_time(read["at"])))
        return 0.5 ** (age_h / HALF_LIFE_H)

    xs = [math.log1p(hours_since_origin(parse_time(r["at"]))) for r in distinct]
    ws = [weight(r) for r in distinct]

    # --- players_tracked: log-linear in log time. ---
    ys_n = [math.log(max(1, r["players_tracked"])) for r in distinct]
    a_n, b_n, sd_n, sxx_n, xbar_n = wls(xs, ys_n, ws)
    x_star = math.log1p(t_target)
    log_n_star = a_n + b_n * x_star
    # The base only grows, and not absurdly.
    log_n_star = min(log_n_star, math.log(n_last) + math.log(MAX_BASE_GROWTH))
    log_n_star = max(log_n_star, math.log(n_last))
    n_star = math.exp(log_n_star)

    # Fit uncertainty for the base, propagated to the target.
    if sxx_n > 0 and len(distinct) > 2:
        lev_n = 1.0 + (x_star - xbar_n) ** 2 / sxx_n
        var_n = sd_n * sd_n * lev_n
    else:
        var_n = 0.04  # a flat history says nothing about the base's growth

    rw_var = (RW_SIGMA * math.sqrt(horizon_days)) ** 2
    # Refresh uncertainty: large while the history is flat (the refresh
    # is still to come and unseen), smaller once several refreshes have
    # been observed.
    refresh_sigma = (
        REFRESH_SIGMA_FLAT if len(distinct) < FLAT_HISTORY_READS else REFRESH_SIGMA_RICH
    )
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        # The 100% trophy: everyone has it, everyone will.
        if last_rate is not None and last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        # Earners per read: rate * base / 100, floored at half a player
        # so a zero rate still fits.
        ys_e = []
        xs_e = []
        ws_e = []
        for j, read in enumerate(distinct):
            rate = read["rates"][i]
            if rate is None:
                continue
            base = read["players_tracked"] if read["players_tracked"] else 1
            earners = max(0.5, rate * base / 100.0)
            ys_e.append(math.log(earners))
            xs_e.append(xs[j])
            ws_e.append(ws[j])
        if not ys_e:
            # No read ever listed this trophy: a tier-based guess, wide.
            med = {"platinum": 0.1, "gold": 0.5, "silver": 1.0}.get(
                (trophy.get("tier") or "").lower(), 2.0
            )
            sigma = math.sqrt(rw_var + refresh_sigma ** 2 + 0.25)
            lg = logit(med)
            rows.append({
                "id": trophy["id"],
                "q": [round4(inv_logit(lg + z * sigma)) for z in Z],
            })
            continue
        a_e, b_e, sd_e, sxx_e, xbar_e = wls(xs_e, ys_e, ws_e)
        log_e_star = a_e + b_e * x_star
        # Earners never fall below the last read's count.
        e_last = max(0.5, (last_rate if last_rate is not None else 0.0) * n_last / 100.0)
        log_e_star = max(log_e_star, math.log(e_last))
        # ... and never above the tier ceiling's share of the base.
        ceil = tier_ceiling(trophy)
        log_e_star = min(log_e_star, math.log(max(0.5, ceil * n_star)))
        e_star = math.exp(log_e_star)
        med = 100.0 * e_star / n_star
        # The median may not move too far from the last read in one jump.
        if last_rate is not None and last_rate > 0:
            move = math.log(max(0.01, med) / max(0.01, last_rate))
            if abs(move) > MAX_LOG_MOVE:
                med = max(0.01, last_rate) * math.exp(
                    MAX_LOG_MOVE if move > 0 else -MAX_LOG_MOVE
                )
        med = min(99.99, max(0.01, med))
        # Flat history: the base has not refreshed in the window, but it
        # will — shift the median down for the dilution that is coming.
        if len(distinct) < FLAT_HISTORY_READS:
            med = min(99.99, max(0.01, med * math.exp(FLAT_HISTORY_SHIFT)))
        # Fit uncertainty for the earners, propagated to the target.
        if sxx_e > 0 and len(xs_e) > 2:
            lev_e = 1.0 + (x_star - xbar_e) ** 2 / sxx_e
            var_e = sd_e * sd_e * lev_e
        else:
            var_e = 0.04
        var = rw_var + refresh_sigma ** 2 + var_n + var_e
        sigma = min(2.0, max(0.35, math.sqrt(var)))
        lg = logit(med)
        qs = [round4(inv_logit(lg + z * sigma)) for z in Z]
        # Keep the row non-decreasing and inside [0, 100].
        qs = [min(100, max(0, q)) for q in qs]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
