#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at each case's target time.

Model (pooled saturation):
  * A single global growth curve is estimated on the logit scale from the
    median trophy at every read, fitted as a + b*ln(1 + hours since
    release). The curve is concave, so growth decelerates as the tracked
    player population saturates.
  * Every trophy rides that curve with its own level offset, estimated
    from its most recent reads.
  * A per-tier ceiling (platinum < gold < silver < bronze) caps how far a
    rare trophy's median can be extrapolated.
  * The forecast spread is measured from how much trophy offsets drift
    around the global curve, scaled by the square root of the horizon,
    with a random-walk floor.

Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (default /output). Deterministic, offline,
Python standard library only.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    """Percent (0-100) -> logit, with the scorer's [0.01, 99.99] clamp."""
    p = clamp(float(x), 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(z):
    """Logit -> percent (0-100), numerically stable."""
    if z >= 0.0:
        return 100.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return 100.0 * e / (1.0 + e)


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


def mad_sigma(xs):
    """Robust sigma estimate from the median absolute deviation."""
    if not xs:
        return 0.0
    med = median(xs)
    return 1.4826 * median([abs(x - med) for x in xs])


def norm_quantile(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996,
         3.754408661907416)
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return num / den
    if p > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    return num / den


def parse_time(s):
    """Parse an ISO-8601 timestamp into an aware UTC datetime, or None."""
    if not isinstance(s, str):
        return None
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def fit_line(xs, ys):
    """Least squares y = a + b*x; returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


# --------------------------------------------------------------------------
# model constants
# --------------------------------------------------------------------------

# Week-one ceilings for the median forecast, in percent, by trophy tier.
TIER_CEILING = {"platinum": 6.0, "gold": 18.0, "silver": 40.0, "bronze": 90.0}
DEFAULT_CEILING = 90.0
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The global curve's logit move from the last read to the target is kept
# inside this band so a wild early fit cannot extrapolate absurdly.
DELTA_MIN, DELTA_MAX = -1.5, 2.5
# A trophy whose offset has been drifting against the global curve (it is
# saturating earlier, or catching up) keeps half of that drift rate over
# the horizon, capped, instead of snapping back to a fixed offset.
DRIFT_KEEP = 0.5
DRIFT_CAP = 0.75
# Random-walk floor for the logit spread: 0.5 per square-root day.
RW_SIGMA_PER_SQRT_DAY = 0.5
SPREAD_MIN, SPREAD_MAX = 0.05, 3.0
OFFSET_WINDOW_H = 6.0


# --------------------------------------------------------------------------
# forecasting
# --------------------------------------------------------------------------

def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [norm_quantile(float(p)) for p in levels]

    # ---- time axis: hours since release midnight (UTC) ----
    origin = None
    rel = (case.get("game") or {}).get("release_date")
    if isinstance(rel, str):
        origin = parse_time(rel if "T" in rel else rel + "T00:00:00Z")

    reads = []
    for h in history:
        if not isinstance(h, dict):
            continue
        t = parse_time(h.get("at")) or parse_time(h.get("slot"))
        rates = h.get("rates") or []
        if t is not None:
            reads.append((t, rates))
    reads.sort(key=lambda r: r[0])

    at_dt = parse_time(case.get("at"))
    if origin is None:
        origin = reads[0][0] if reads else (at_dt or datetime(2026, 11, 19, tzinfo=timezone.utc))

    def hours(dt):
        return max(0.0, (dt - origin).total_seconds() / 3600.0)

    target_dt = parse_time(case.get("target"))
    if target_dt is None and at_dt is not None and case.get("horizon_minutes") is not None:
        target_dt = at_dt + timedelta(minutes=float(case["horizon_minutes"]))
    if target_dt is None:
        target_dt = at_dt or (reads[-1][0] if reads else origin)

    t_target = hours(target_dt)
    t_at = hours(at_dt) if at_dt is not None else (hours(reads[-1][0]) if reads else t_target)
    horizon_h = max(0.0, t_target - t_at)

    # ---- per-trophy logit series ----
    n_tr = len(trophies)
    series = [[] for _ in range(n_tr)]          # (t_hours, logit) per trophy
    g_times, g_vals = [], []                    # median logit per read
    for t, rates in reads:
        th = hours(t)
        lg_all = []
        for i in range(min(n_tr, len(rates))):
            r = rates[i]
            if r is None:
                continue
            try:
                lg = logit_pct(r)
            except (TypeError, ValueError):
                continue
            series[i].append((th, lg))
            lg_all.append(lg)
        if lg_all:
            g_times.append(th)
            g_vals.append(median(lg_all))

    # ---- global growth curve: median logit vs ln(1 + hours) ----
    if g_vals:
        a, b = fit_line([math.log1p(x) for x in g_times], g_vals)
    else:
        a, b = logit_pct(1.0), 0.0

    def ghat(t_h):
        return a + b * math.log1p(max(0.0, t_h))

    t_last = g_times[-1] if g_times else t_at
    delta = clamp(ghat(t_target) - ghat(t_last), DELTA_MIN, DELTA_MAX)
    g_target = ghat(t_last) + delta

    # ---- per-trophy offsets from recent reads ----
    def window_offset(points, t_ref, window_h, from_start=False):
        """Mean (logit - global curve) over a window of a trophy's reads."""
        if not points:
            return None
        if from_start:
            sel = [p for p in points if p[0] <= points[0][0] + window_h]
            if len(sel) < 3:
                sel = points[:3]
        else:
            sel = [p for p in points if p[0] >= t_ref - window_h]
            if len(sel) < 3:
                sel = points[-3:]
        return sum(lg - ghat(t) for t, lg in sel) / len(sel)

    offsets = []
    last_rates = []
    for pts in series:
        offsets.append(window_offset(pts, t_last, OFFSET_WINDOW_H))
        last_rates.append(inv_logit_pct(pts[-1][1]) if pts else None)

    # ---- spread: offset drift around the global curve, horizon-scaled ----
    # Also record each trophy's offset drift rate (logits per hour) so the
    # forecast offset can keep part of it instead of assuming it stops.
    drift_vs = []
    drift_rates = [0.0] * n_tr
    residuals = []
    for i, pts in enumerate(series):
        if not pts:
            continue
        full_off = sum(lg - ghat(t) for t, lg in pts) / len(pts)
        residuals.extend(lg - ghat(t) - full_off for t, lg in pts)
        if len(pts) >= 4:
            span = pts[-1][0] - pts[0][0]
            if span >= 1.0:
                early = window_offset(pts, t_last, OFFSET_WINDOW_H, from_start=True)
                late = offsets[i]
                if early is not None and late is not None:
                    drift_vs.append((late - early) / math.sqrt(span))
                    if span >= 6.0:
                        drift_rates[i] = (late - early) / span

    s_drift = mad_sigma(drift_vs) * math.sqrt(horizon_h) if len(drift_vs) >= 5 else 0.0
    s_resid = mad_sigma(residuals)
    s_floor = RW_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_h / 24.0)
    spread = clamp(max(s_drift, s_resid, s_floor), SPREAD_MIN, SPREAD_MAX)

    # ---- fallback level for trophies with no reads: tier-typical logit ----
    tier_logits = {}
    all_logits = []
    for i, tr in enumerate(trophies):
        if series[i]:
            lg = series[i][-1][1]
            tier_logits.setdefault(str(tr.get("tier", "")).lower(), []).append(lg)
            all_logits.append(lg)
    global_fallback = median(all_logits) if all_logits else logit_pct(1.0)

    def tier_fallback_logit(tier):
        vals = tier_logits.get(tier)
        return median(vals) if vals else global_fallback

    # ---- build the forecast rows ----
    rows = []
    for i, tr in enumerate(trophies):
        tier = str(tr.get("tier", "")).lower()
        ceiling = TIER_CEILING.get(tier, DEFAULT_CEILING)
        last_rate = last_rates[i]
        spread_i = spread
        if offsets[i] is None:
            base = tier_fallback_logit(tier)
            med_logit = base + delta
            if last_rate is None:
                last_rate = inv_logit_pct(base)
            spread_i = spread * 1.5
        else:
            drift_adj = clamp(DRIFT_KEEP * drift_rates[i] * horizon_h,
                              -DRIFT_CAP, DRIFT_CAP)
            med_logit = g_target + offsets[i] + drift_adj

        # tier ceiling on the median (never below the current level)
        cap = max(ceiling, last_rate if last_rate is not None else 0.0)
        rate50 = min(inv_logit_pct(med_logit), cap)
        med_logit = logit_pct(rate50)

        # quantiles may exceed the median ceiling but not absurdly
        qcap = min(99.99, max(3.0 * ceiling, last_rate or 0.0, rate50))
        q = []
        for z in zs:
            v = inv_logit_pct(med_logit + z * spread_i)
            q.append(round(clamp(v, 0.01, qcap), 4))
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr.get("id", i + 1), "q": q})

    return {"trophies": rows}


def fallback_forecast(case):
    """Persistence with a random-walk spread; used only if the model fails."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    levels = case.get("levels") or DEFAULT_LEVELS
    zs = [norm_quantile(float(p)) for p in levels]
    horizon_days = 1.0
    at_dt = parse_time(case.get("at"))
    target_dt = parse_time(case.get("target"))
    if at_dt is not None and target_dt is not None:
        horizon_days = max(0.0, (target_dt - at_dt).total_seconds() / 86400.0)
    elif case.get("horizon_minutes") is not None:
        horizon_days = float(case["horizon_minutes"]) / 1440.0
    s = RW_SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)

    last = {}
    for h in history:
        if not isinstance(h, dict):
            continue
        for i, r in enumerate(h.get("rates") or []):
            if r is not None:
                last[i] = r
    rows = []
    for i, tr in enumerate(trophies):
        center = logit_pct(last.get(i, 1.0))
        q = [round(clamp(inv_logit_pct(center + z * s), 0.01, 99.99), 4) for z in zs]
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": tr.get("id", i + 1), "q": q})
    return {"trophies": rows}


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        stem = name[:-3]
        case = None
        try:
            with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
                case = json.load(f)
            out = forecast_case(case)
        except Exception:
            if case is None:
                continue
            try:
                out = fallback_forecast(case)
            except Exception:
                continue
        text = json.dumps(out, separators=(",", ":"))
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
