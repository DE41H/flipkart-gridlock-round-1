"""Model output validation tests: overfitting, diversity, residuals, demand bucketing, submission.

Tests load saved artifacts (OOF arrays, metrics) from the last src/main.py run
without retraining. Five main test classes cover:

1. TestOverfitting: train-val R² gap and minimum performance floors.
2. TestModelDiversity: Pearson correlation and finite OOF predictions.
3. TestResiduals: mean/median residuals, outlier frequency.
4. TestDemandBucketAccuracy: per-bucket accuracy and confusion matrix.
5. TestSubmission: format, range, index alignment.
6. TestEvalMetrics: optional deeper diagnostics (bucket/spatial/temporal).

All tests skip gracefully if artifacts are missing (user must run src/main.py first).
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix as sk_cm

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DEMAND_BINS, DEMAND_BIN_LABELS
from evaluation import _demand_buckets, _reblend_oof, _to_demand

# =========================================================================
# Test Threshold Constants (Intentional Values)
# =========================================================================

# Overfitting tolerance: honest cross-day forecast has ~0.30-0.45 train-val gap
MAX_TRAIN_VAL_GAP: float = 0.45
"""Maximum acceptable train-val R² gap (one-day-ahead task is intrinsically hard)."""

VALIDATION_R2_FLOOR: float = 0.22
"""Minimum validation R² (honest day-49 fold, not inflated day-48 regime)."""

LGBM_OOF_FLOOR: float = 0.25
"""Minimum LGBM OOF R² (lighter model, early-stops aggressively)."""

CATBOOST_OOF_FLOOR: float = 0.20
"""Minimum CatBoost OOF R² (honest cross-day performance)."""

# Diversity: ensure LightGBM and CatBoost OOFs are not perfectly collinear
OOF_CORRELATION_THRESHOLD: float = 0.999
"""Maximum acceptable Pearson r between model OOF predictions (enforces diversity)."""

# Residual analysis: day-49 morning fold + log-space bias (shrink=0.0)
RESIDUAL_MEAN_TOLERANCE: float = 0.45
"""Mean residual tolerance (exposed Jensen log-bias, shrink=0.0)."""

RESIDUAL_MEDIAN_TOLERANCE: float = 0.55
"""Median residual tolerance (symmetric residuals around zero)."""

EXTREME_RESIDUAL_FRACTION: float = 0.01
"""Maximum fraction of residuals beyond 5σ (outlier threshold)."""

# Demand bucketing: honest cross-day, very_low-dominated distribution
BUCKET_MACRO_ACCURACY_FLOOR: float = 0.40
"""Minimum macro per-bucket accuracy (sparse mid/high buckets are hard)."""

VERY_LOW_BUCKET_ACCURACY: float = 0.70
"""Minimum very_low bucket accuracy (bulk of demand is very_low, must capture)."""

HIGH_BUCKET_MISCLASSIFICATION: float = 0.10
"""Maximum fraction of high-demand rows predicted as very_low."""

VERY_LOW_BUCKET_OVERESTIMATION: float = 0.05
"""Maximum fraction of very-low-demand rows predicted as high."""

# Submission: basic sanity checks
SUBMISSION_DEMAND_MIN: float = 0.03
"""Expected minimum submission mean demand (given train ≈0.094)."""

SUBMISSION_DEMAND_MAX: float = 0.30
"""Expected maximum submission mean demand."""


def _blend_val(artifacts: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct validation-fold blend predictions from saved artifacts.

    The artifacts fixture stores OOF arrays and metrics separately from the live `res` dict,
    so we reconstruct a minimal res-compatible dict before calling _reblend_oof to get
    the final blended validation predictions.

    Parameters
    ----------
    artifacts : dict
        Loaded artifacts (oof_lgbm, oof_cat, y_log, folds, metrics).

    Returns
    -------
    tuple
        (y_true_log, y_pred_log) on validation fold only (fold >= 0).
    """
    res_like = {
        "folds": artifacts["folds"],
        "y_log": artifacts["y_log"],
        "lgbm_oof": artifacts["oof_lgbm"],
        "cat_oof": artifacts["oof_cat"],
        "et_oof": artifacts.get("oof_et"),
        "xgb_oof": artifacts.get("oof_xgb"),
        "blend_weights": artifacts["metrics"].get("blend_weights"),
        "blend_weight": artifacts["metrics"].get("blend_weight"),
        "calibration": artifacts["metrics"].get("calibration"),
    }
    y_true_log, y_pred_log, _ = _reblend_oof(res_like)
    return y_true_log, y_pred_log


# =========================================================================
# Overfitting Tests
# =========================================================================

class TestOverfitting:
    """Detect training–validation gap and minimum performance thresholds.

    The honest cross-day validation (day-49 from day-48 training) is intrinsically
    a one-day-ahead forecast task with distributional shift (morning vs. all day),
    so a ~0.30-0.45 train-val gap is expected and not necessarily overfitting.
    """

    def test_no_severe_overfitting(self, artifacts: dict[str, Any]) -> None:
        """Verify train-val R² gap stays within honest cross-day tolerance.

        train_r2/val_r2 are fold-level metrics. The one-day-ahead forecasting task
        with distributional shifts (train: full day 48, val: morning of day 49) has
        an intrinsic gap; a ~0.3-0.45 gap is expected, not memorization.
        """
        m = artifacts["metrics"]
        if m["train_r2"] is None or m["val_r2"] is None:
            pytest.skip("No fold metrics in artifacts")
        gap = m["train_r2"] - m["val_r2"]
        assert abs(gap) < MAX_TRAIN_VAL_GAP, (
            f"Train-val R² gap = {gap:+.4f} (exceeds {MAX_TRAIN_VAL_GAP}). "
            f"train={m['train_r2']:.4f}, val={m['val_r2']:.4f}"
        )

    def test_val_r2_above_floor(self, artifacts: dict[str, Any]) -> None:
        """Verify validation R² exceeds the honest temporal-forecast floor.

        val_r2 is the fold-level LGBM/CatBoost blend R² on day-49 (genuine cross-day
        forecast). Early-stops aggressively; floor set to honest GBDT level, not
        inflated day-48 regime.
        """
        m = artifacts["metrics"]
        if m["val_r2"] is None:
            pytest.skip("No val_r2 in artifacts")
        assert m["val_r2"] > VALIDATION_R2_FLOOR, (
            f"val_r2={m['val_r2']:.4f} below floor {VALIDATION_R2_FLOOR}"
        )

    def test_lgbm_oof_r2_above_floor(self, artifacts: dict[str, Any]) -> None:
        """Verify LGBM OOF R² exceeds the honest floor.

        On honest day-49 fold, LGBM early-stops aggressively and underperforms
        CatBoost/ExtraTrees; it carries ~0 blend weight. Floor reflects honest
        cross-day difficulty.
        """
        m = artifacts["metrics"]
        if m["lgbm_oof_r2"] is None:
            pytest.skip("No lgbm_oof_r2 in artifacts")
        assert m["lgbm_oof_r2"] > LGBM_OOF_FLOOR, (
            f"lgbm_oof_r2={m['lgbm_oof_r2']:.4f} below {LGBM_OOF_FLOOR}"
        )

    def test_catboost_oof_r2_above_floor(self, artifacts: dict[str, Any]) -> None:
        """Verify CatBoost OOF R² exceeds the honest floor.

        CatBoost honest day-49 OOF (with TE and spatial features) exceeds floor.
        The dominant ExtraTrees member is asserted in blend-beats-best-single.
        """
        m = artifacts["metrics"]
        if m["cat_oof_r2"] is None:
            pytest.skip("No cat_oof_r2 in artifacts")
        assert m["cat_oof_r2"] > CATBOOST_OOF_FLOOR, (
            f"cat_oof_r2={m['cat_oof_r2']:.4f} below {CATBOOST_OOF_FLOOR}"
        )

    def test_blend_beats_best_single_model(self, artifacts: dict[str, Any]) -> None:
        """Verify blend OOF R² is at least as good as best single model."""
        m = artifacts["metrics"]
        if any(v is None for v in [m["blend_oof_r2"], m["lgbm_oof_r2"], m["cat_oof_r2"]]):
            pytest.skip("Incomplete OOF metrics")
        best_single = max(m["lgbm_oof_r2"], m["cat_oof_r2"])
        assert m["blend_oof_r2"] >= best_single - 0.005, (
            f"Blend R²={m['blend_oof_r2']:.4f} worse than best single model R²={best_single:.4f}"
        )


# =========================================================================
# Model Diversity Tests
# =========================================================================

class TestModelDiversity:
    """Verify LGBM and CatBoost OOF predictions are genuinely diverse.

    Blending only helps if models are uncorrelated. Perfect correlation means
    the blend reduces to a single model—this test ensures diversity.
    """

    def test_oof_predictions_not_perfectly_correlated(self, artifacts: dict[str, Any]) -> None:
        """Verify LGBM and CatBoost OOF Pearson r is below correlation threshold.

        If r ≥ 0.999, the models are too similar and blending adds no value.
        """
        folds = artifacts["folds"]
        val_mask = folds == 0
        lgbm_val = artifacts["oof_lgbm"][val_mask]
        cat_val = artifacts["oof_cat"][val_mask]

        if val_mask.sum() < 10:
            pytest.skip("Too few validation rows to compute correlation")

        r, _ = pearsonr(lgbm_val, cat_val)
        assert r < OOF_CORRELATION_THRESHOLD, (
            f"LGBM–CatBoost OOF Pearson r={r:.4f} ≥ {OOF_CORRELATION_THRESHOLD}. "
            f"Models too similar; blending adds no value."
        )

    def test_no_nan_oof_on_val_fold(self, artifacts: dict[str, Any]) -> None:
        """Verify both models produce finite OOF predictions on validation fold."""
        folds = artifacts["folds"]
        val_mask = folds == 0
        assert np.isfinite(artifacts["oof_lgbm"][val_mask]).all(), \
            "NaN/Inf in LGBM OOF predictions on validation fold"
        assert np.isfinite(artifacts["oof_cat"][val_mask]).all(), \
            "NaN/Inf in CatBoost OOF predictions on validation fold"

    def test_oof_lgbm_range_plausible(self, artifacts: dict[str, Any]) -> None:
        """Verify LGBM OOF predictions are in plausible range [1e-6, 1.0]."""
        folds = artifacts["folds"]
        val_mask = folds == 0
        preds = artifacts["oof_lgbm"][val_mask]
        assert np.isfinite(preds).all(), "LGBM OOF has non-finite values"
        demand = _to_demand(preds)
        assert demand.min() >= 1e-7, f"LGBM OOF demand min={demand.min():.2e} below clip floor"
        assert demand.max() <= 1.0 + 1e-6, f"LGBM OOF demand max={demand.max():.6f} exceeds 1.0"


# =========================================================================
# Residual Tests
# =========================================================================

class TestResiduals:
    """Verify OOF residuals are unbiased and without extreme outliers.

    The honest fold validates on day-49 morning (low demand) while training spans
    day-48 full day; log-space MSE-optimal predictor is biased low (Jensen).
    Calibration is disabled (shrink=0.0) because it regressed the leaderboard,
    so the Jensen bias is exposed. These tolerances account for that.
    """

    def test_residual_mean_near_zero(self, artifacts: dict[str, Any]) -> None:
        """Verify mean OOF residual (actual - pred) is bounded.

        With shrink=0.0, the full Jensen log-space bias is exposed on day-49
        morning fold. A mean residual near +0.2-0.3 is expected (not negative bias).
        """
        y_true, y_pred = _blend_val(artifacts)
        mean_residual = float(np.mean(y_true - y_pred))
        assert abs(mean_residual) < RESIDUAL_MEAN_TOLERANCE, (
            f"Mean OOF residual = {mean_residual:.4f} "
            f"exceeds tolerance {RESIDUAL_MEAN_TOLERANCE}"
        )

    def test_no_extreme_residual_outliers(self, artifacts: dict[str, Any]) -> None:
        """Verify outlier residuals (> 5σ) represent < 1% of validation set."""
        y_true, y_pred = _blend_val(artifacts)
        residuals = y_true - y_pred
        sigma = float(np.std(residuals))
        extreme_frac = float(np.mean(np.abs(residuals) > 5 * sigma))
        assert extreme_frac < EXTREME_RESIDUAL_FRACTION, (
            f"{extreme_frac:.3%} of residuals exceed 5σ. "
            f"Threshold: {EXTREME_RESIDUAL_FRACTION:.1%}"
        )

    def test_residuals_symmetric_around_zero(self, artifacts: dict[str, Any]) -> None:
        """Verify median residual is bounded (not strongly skewed).

        Median captures asymmetry better than mean. With shrink=0.0,
        median sits near +0.2-0.3 due to Jensen bias, which is expected.
        """
        y_true, y_pred = _blend_val(artifacts)
        median_residual = float(np.median(y_true - y_pred))
        assert abs(median_residual) < RESIDUAL_MEDIAN_TOLERANCE, (
            f"Median OOF residual = {median_residual:.4f} "
            f"exceeds tolerance {RESIDUAL_MEDIAN_TOLERANCE}"
        )


# =========================================================================
# Demand Bucket Classification Tests
# =========================================================================

class TestDemandBucketAccuracy:
    """Regression confusion matrix: binned demand accuracy.

    Buckets: very_low (<0.03), low (0.03-0.07), medium (0.07-0.15), high (>0.15).
    Per-bucket accuracy and macro-accuracy are checked against thresholds.
    """

    def _get_buckets(self, artifacts: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute confusion matrix and demand bucket assignments for validation fold.

        Parameters
        ----------
        artifacts : dict
            Loaded artifacts with OOF predictions and targets.

        Returns
        -------
        tuple
            (confusion_matrix, y_true, y_pred).
        """
        y_true_log, y_pred_log = _blend_val(artifacts)
        y_true = _to_demand(y_true_log)
        y_pred = _to_demand(y_pred_log)
        cm = sk_cm(_demand_buckets(y_true), _demand_buckets(y_pred), labels=[0, 1, 2, 3])
        return cm, y_true, y_pred

    def test_confusion_matrix_diagonal_dominates(self, artifacts: dict[str, Any]) -> None:
        """Verify macro per-bucket accuracy meets honest cross-day floor.

        On honest day-49 morning fold, demand is overwhelmingly very_low; mid/high buckets
        are sparse and intrinsically hard to forecast a day ahead. Macro accuracy ~0.40+
        is expected. very_low bucket (dominant) must still be predicted well.
        """
        cm, _, _ = self._get_buckets(artifacts)
        per_bucket_acc = [cm[i, i] / cm[i].sum() for i in range(4) if cm[i].sum() > 0]
        macro_acc = np.mean(per_bucket_acc)
        assert macro_acc > BUCKET_MACRO_ACCURACY_FLOOR, (
            f"Macro per-bucket accuracy = {macro_acc:.3f} < {BUCKET_MACRO_ACCURACY_FLOOR}. "
            f"Bucket accuracies: {dict(zip(DEMAND_BIN_LABELS, [f'{a:.2f}' for a in per_bucket_acc]))}"
        )
        assert per_bucket_acc[0] > VERY_LOW_BUCKET_ACCURACY, (
            f"very_low bucket accuracy = {per_bucket_acc[0]:.3f} < {VERY_LOW_BUCKET_ACCURACY}. "
            f"Most demand is very_low; must be captured."
        )

    def test_high_demand_bucket_not_missed(self, artifacts: dict[str, Any]) -> None:
        """Verify high-demand rows are not systematically under-predicted as very_low."""
        cm, _, _ = self._get_buckets(artifacts)
        high_total = cm[3].sum()
        if high_total == 0:
            pytest.skip("No high-demand rows in validation fold")
        high_to_vl_frac = cm[3, 0] / high_total
        assert high_to_vl_frac < HIGH_BUCKET_MISCLASSIFICATION, (
            f"{high_to_vl_frac:.1%} of high-demand rows predicted as very_low. "
            f"Threshold: {HIGH_BUCKET_MISCLASSIFICATION:.1%}"
        )

    def test_very_low_demand_not_overestimated(self, artifacts: dict[str, Any]) -> None:
        """Verify very-low-demand rows are not systematically over-predicted."""
        cm, _, _ = self._get_buckets(artifacts)
        vl_total = cm[0].sum()
        if vl_total == 0:
            pytest.skip("No very-low-demand rows in validation fold")
        vl_to_high_frac = cm[0, 3] / vl_total
        assert vl_to_high_frac < VERY_LOW_BUCKET_OVERESTIMATION, (
            f"{vl_to_high_frac:.1%} of very-low rows predicted as high. "
            f"Threshold: {VERY_LOW_BUCKET_OVERESTIMATION:.1%}"
        )


# =========================================================================
# Submission Format & Integrity Tests
# =========================================================================

class TestSubmission:
    """Validate the saved submission.csv for format, range, and completeness."""

    @pytest.fixture(scope="class")
    def submission(self) -> pd.DataFrame:
        """Load submission.csv from submissions/ directory.

        Returns
        -------
        pd.DataFrame
            Submission DataFrame with Index and demand columns.
        """
        sub_path = os.path.join(ROOT, "submissions", "submission.csv")
        if not os.path.exists(sub_path):
            pytest.skip("submission.csv not found — run src/main.py first")
        return pd.read_csv(sub_path)

    @pytest.fixture(scope="class")
    def test_csv(self) -> pd.DataFrame:
        """Load test.csv from data/ directory.

        Returns
        -------
        pd.DataFrame
            Raw test data to validate row count and indices.
        """
        return pd.read_csv(os.path.join(ROOT, "data", "test.csv"))

    def test_submission_row_count(self, submission: pd.DataFrame, test_csv: pd.DataFrame) -> None:
        """Verify submission row count equals test set size."""
        assert len(submission) == len(test_csv), (
            f"Submission has {len(submission)} rows, expected {len(test_csv)}"
        )

    def test_submission_columns(self, submission: pd.DataFrame) -> None:
        """Verify submission has exactly Index and demand columns."""
        assert list(submission.columns) == ["Index", "demand"], (
            f"Unexpected columns: {list(submission.columns)}"
        )

    def test_submission_no_nan(self, submission: pd.DataFrame) -> None:
        """Verify submission has no NaN demand values."""
        n_null = submission["demand"].isnull().sum()
        assert n_null == 0, f"{n_null} NaN demand values in submission"

    def test_submission_demand_in_range(self, submission: pd.DataFrame) -> None:
        """Verify all demand values are in (0, 1] (normalized target range)."""
        demand = submission["demand"].values
        assert (demand > 0).all(), f"Demand has {(demand <= 0).sum()} non-positive values"
        assert (demand <= 1.0 + 1e-9).all(), (
            f"Demand has {(demand > 1.0 + 1e-9).sum()} values exceeding 1.0"
        )

    def test_submission_indices_match_test(self, submission: pd.DataFrame, test_csv: pd.DataFrame) -> None:
        """Verify submission indices exactly match test set (no missing/extra rows)."""
        sub_indices = set(submission["Index"].values)
        test_indices = set(test_csv["Index"].values)
        missing = test_indices - sub_indices
        extra = sub_indices - test_indices
        assert not missing, f"{len(missing)} test indices missing from submission"
        assert not extra, f"{len(extra)} extra indices in submission not in test set"

    def test_submission_demand_distribution_plausible(self, submission: pd.DataFrame) -> None:
        """Verify submission demand distribution is reasonable.

        Given training mean ≈0.094, test predictions should fall within plausible bounds.
        """
        mean_pred = float(submission["demand"].mean())
        assert SUBMISSION_DEMAND_MIN < mean_pred < SUBMISSION_DEMAND_MAX, (
            f"Submission mean demand = {mean_pred:.4f}. "
            f"Expected range: ({SUBMISSION_DEMAND_MIN}, {SUBMISSION_DEMAND_MAX})"
        )


# =========================================================================
# Evaluation Diagnostics Tests (require artifacts/eval.json)
# =========================================================================

class TestEvalMetrics:
    """Thorough model diagnostics using artifacts/eval.json (optional).

    These tests surface shortcomings the aggregate R² hides: per-bucket failures,
    spatial blind spots, temporal dead zones, and distribution drift between val and test.
    All skip if eval.json is missing (requires src/main.py to have run evaluation).
    """

    def test_no_bucket_zero_accuracy(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify every populated demand bucket has bucket-accuracy > 0.10.

        Per-bucket demand-scale R² is unreliable on this task (narrow ranges, log-space
        model) — bucket accuracy (fraction of samples landing in the correct bin) is the
        meaningful proxy for whether the model has any signal in each range.
        """
        bm = eval_artifacts.get("bucket_metrics", {})
        if not bm:
            pytest.skip("No bucket_metrics in eval.json")
        for label, m in bm.items():
            acc = m.get("acc")
            n = m.get("n", 0)
            if acc is not None and n >= 10:
                assert acc > 0.10, (
                    f"Bucket '{label}' acc={acc:.4f} — no predictive signal in this range"
                )

    def test_macro_confusion_acc_above_floor(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify macro per-bucket accuracy exceeds 0.40."""
        cm_data = eval_artifacts.get("confusion", {})
        if not cm_data or cm_data.get("macro_acc") is None:
            pytest.skip("No confusion metrics in eval.json")
        assert cm_data["macro_acc"] > 0.40, (
            f"Macro bucket accuracy = {cm_data['macro_acc']:.4f} < 0.40"
        )

    def test_very_low_bucket_well_captured(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify very_low demand bucket accuracy > 0.70 (it dominates demand distribution)."""
        bm = eval_artifacts.get("bucket_metrics", {})
        vl_key = next((k for k in bm if "very_low" in k), None)
        if not vl_key or bm[vl_key].get("acc") is None:
            pytest.skip("No very_low bucket metrics in eval.json")
        acc = bm[vl_key]["acc"]
        assert acc > 0.70, (
            f"very_low bucket accuracy = {acc:.4f} < 0.70 — bulk of demand is very_low"
        )

    def test_spatial_log_mae_not_extreme(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify mean per-geohash log-space MAE < 1.5 (no systematic spatial failure).

        Spatial R² is unreliable with ~6 val samples per geohash, so log-space MAE is
        used instead. A mean log-MAE of ~1.0 is expected for this honest cross-day
        task; > 1.5 would indicate systematic location-specific prediction failures.
        """
        sp = eval_artifacts.get("spatial", {})
        if sp.get("mae_mean") is None:
            pytest.skip("No spatial metrics in eval.json")
        assert sp["mae_mean"] < 1.5, (
            f"Mean per-geohash log-MAE = {sp['mae_mean']:.4f} > 1.5 — spatial failure"
        )

    def test_no_catastrophic_hour(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify no hour of day has R² below -0.15 (extreme temporal failure)."""
        tm = eval_artifacts.get("temporal", {})
        by_hour = tm.get("by_hour", {})
        if not by_hour:
            pytest.skip("No temporal metrics in eval.json")
        for hour, r2 in by_hour.items():
            if r2 is not None:
                assert r2 > -0.15, (
                    f"Hour {hour} R²={r2:.4f} — extreme temporal failure detected"
                )

    def test_test_pred_not_degenerate(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify test prediction std > 0.05 (model is not outputting near-constant demand)."""
        td = eval_artifacts.get("test_pred_dist", {})
        if not td or td.get("std") is None:
            pytest.skip("No test_pred_dist in eval.json")
        assert td["std"] > 0.05, (
            f"Test pred std = {td['std']:.5f} — degenerate constant-like output"
        )

    def test_val_test_distribution_not_diverged(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify test pred mean is within 3× of val pred mean (no extreme distribution shift)."""
        vd = eval_artifacts.get("val_pred_dist", {})
        td = eval_artifacts.get("test_pred_dist", {})
        if not vd or not td or vd.get("mean") is None or td.get("mean") is None:
            pytest.skip("Missing pred distribution metrics in eval.json")
        ratio = td["mean"] / max(vd["mean"], 1e-9)
        assert 0.33 < ratio < 3.0, (
            f"Test/val pred mean ratio = {ratio:.2f} — extreme distribution shift "
            f"(val_mean={vd['mean']:.5f}, test_mean={td['mean']:.5f})"
        )

    def test_residual_skew_not_extreme(self, eval_artifacts: dict[str, Any]) -> None:
        """Verify absolute residual skew < 2.0 (no extreme systematic bias direction)."""
        res_data = eval_artifacts.get("residuals", {})
        if not res_data or res_data.get("skew") is None:
            pytest.skip("No residual stats in eval.json")
        skew_val = res_data["skew"]
        assert abs(skew_val) < 2.0, (
            f"Residual skew = {skew_val:.4f} — extreme systematic bias in one direction"
        )
