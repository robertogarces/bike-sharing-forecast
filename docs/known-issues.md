# Known Issues & Technical Debt

Findings from a code review of the prediction/training pipeline (2026-06-28),
triggered by the discovery of the backfill feature-reuse bug. None of the items
below are currently producing incorrect results the way that bug was, but they
are worth hardening. Ordered by severity.

---

## 🟠 Medium — No rollback if promotion fails mid-loop in `retrain.py`

**Where:** `src/bike_sharing/models/retrain.py` (`main` → `evaluate_promotion_combinations` → `choose_best_combination` → `promote_best_combination`)

Promotion no longer decides `registered`/`casual` independently — it evaluates
all four combinations of (registered, casual) × (new, production) using actual
summed and clipped predictions on the same `val.csv`, and promotes whichever
combination has the lowest combined RMSE (defaulting to keeping the current
pair on a tie). Since the current production pair is always one of the four
evaluated candidates, this design means promotion **can never choose a worse
combination than what's already deployed** — a mixed pair (e.g. new
`registered` + old `casual`) is now an intentional, RMSE-validated outcome, not
an accidental risk.

The residual risk is narrower: `promote_best_combination`'s loop sets the
`production` alias for whichever slot(s) the chosen combination requires. If
this loop fails partway through (MLflow network error, timeout) after
re-aliasing `registered` but before `casual`, production ends up with a pair
that was **never one of the four evaluated/validated combinations** — the one
scenario the RMSE comparison doesn't protect against, because it happens after
the comparison already picked a winner.

**Why it matters:** same "silent degradation" failure mode as the backfill bug.
Retraining is weekly, so a desync would take a while to notice. Note that as of
`b68b1bb` (see Resolved below), a genuine MLflow failure during the
*evaluation* stage now propagates loudly instead of being silently swallowed —
this entry is specifically about a failure during the *alias-setting* stage,
which remains unguarded.

**Suggested fix:** apply both aliases atomically, or roll back the first if the
second fails. Add a regression test for the partial-failure path specifically.

---

## 🟡 Medium-low — `days_since_start` coupled to `min_date`

**Where:** `src/bike_sharing/features/build_features.py:104`, `src/bike_sharing/models/predict.py`

`days_since_start = (date - min_date).days`. Training uses
`min_date = df["dteday"].min()`; inference uses `min_date = past["dteday"].min()`.
These only stay consistent while `hour_past.csv` keeps its earliest record. If a
sliding window is ever introduced (trimming old data), `min_date` would shift and
this high-MI feature (MI 0.15) would silently skew between train and inference.

Additionally, `days_since_start` grows monotonically: at inference it is always
larger than anything seen in training, and LightGBM does not extrapolate (it
saturates at the last split). Weekly retraining refreshes the range, but between
retrains the "growth" component flattens. This is a known limitation of trend
features, not a bug — worth documenting.

---

## 🟡 Low / latent — Inference features use the *current* hour as a proxy

**Where:** `src/bike_sharing/models/predict.py` (`build_next_hour_features`)

The next-hour row is copied from the current hour and `workingday`, `season`, and
weather (`temp`, `hum`, `weathersit`) are **not** updated to the target hour:

```python
next_row["hr_x_season"] = next_hr * current["season"]   # season of H-1, not H
next_row["temp"] = ...  # inherited from current (H-1)
```

- **workingday / season:** only differ at the midnight (date) boundary, where
  `hr=0` zeroes the products (`hr_workday`, `hr_weekend`, `hr_x_season`). So today
  this is **masked and harmless**. It is latent: if `workingday` or `season` were
  ever added as standalone features (not multiplied by `hr`), it would break
  silently.
- **Weather:** uses H-1's weather as a proxy for H. Training uses H's actual
  weather, so there is a real train/inference difference — but it is inherent to
  forecasting (future weather is unknown without a weather forecast input). A
  legitimate design limitation; document it.

---

## 🟡 Low — Positional lags with gapped data

**Where:** `src/bike_sharing/features/build_features.py:40`

`df["cnt"].shift(lag)` is a **positional** shift. The UCI dataset has missing
hours (confirmed: 2 gaps in the live era), so `cnt_lag_24` is not exactly "same
hour yesterday" near gaps. This is **not** a train/inference skew (both paths are
positional and consistent), but it degrades the feature's meaning around gaps.

---

## ⚪ Cosmetic (no effect on results)

- **`train.py` saves the models twice** — the `booster_.save_model(...)` block is
  duplicated. Redundant, harmless.
- **`drop_cols` in `configs/features/default.yaml`** lists `cnt_lag_9` and
  `cnt_lag_10`, which are never created — no-op, a sign of config drift.
- **`mnth_sin` / `mnth_cos`** are created in `build_calendar_features` then
  dropped immediately, and are not in `FEATURES` — dead code.
- **`build_features.py:169`** the "Calendar features" comment is unindented
  (column 0) inside `main()` — cosmetic.

---

## Resolved

- **Backfill feature-reuse bug** (fixed 2026-06-28): the main next-hour
  prediction reused `X`/`next_row` overwritten inside the backfill loop, producing
  a duplicate of the current hour with the wrong `hr`. Fixed with loop-local
  variables; corrupted live-era predictions regenerated; regression test added
  (`tests/test_predict.py::test_run_main_prediction_uses_next_hour_not_backfill`).

- **Transient MLflow errors mistaken for bootstrap** (fixed in `b68b1bb`):
  `get_production_baseline_rmse()` and `evaluate_promotion_combinations()` in
  `retrain.py` caught bare `MlflowException`, treating *any* error — including
  transient/auth/network failures — the same as "no production model exists
  yet." This could silently disable the performance-degradation gate, or
  promote both new models unconditionally with no real comparison against
  production. Fixed by checking `e.error_code == RESOURCE_DOES_NOT_EXIST`
  specifically; every other MLflow exception now propagates and fails the job.

- **Drift reference snapshot desync on casual-only promotions** (fixed in
  `b68b1bb`): the drift reference snapshot (a copy of drift-relevant training
  features, attached as an MLflow artifact at promotion time so the drift
  reference reflects what production actually learned from) was briefly
  changed to target whichever model actually promoted (registered or casual).
  But `drift_detection.py::_load_reference_snapshot()` only ever reads from
  `{project}-registered`'s production run — so a casual-only promotion wrote
  the snapshot somewhere never read, silently falling back to the stale/live
  reference. Fixed by always targeting the registered run's artifact,
  regardless of which model(s) actually promoted.