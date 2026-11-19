# Pooled Saturation Forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hayunkarimun/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hayunkarimun/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players earn each trophy in week one, for [make.fun](https://make.fun)'s [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) bounty.

`main.py` (Python 3.13, standard library only) reads each case in the input directory and writes five quantiles (5%, 25%, 50%, 75%, 95%) of every trophy's earn rate at the case target. The model is pooled: it fits one global growth curve on the logit scale (median rate across trophies against log time since release), gives each trophy its own offset from recent reads with a damped drift, caps medians by per-tier ceilings, and sets the spread from cross-trophy residual drift scaled by the square root of the horizon. Deterministic, offline, runs in well under a second.

Run it:

    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

New to make.fun? Join with referral code `4pxjyxse`: https://make.fun/r/4pxjyxse
