"""Feature engineering tests: leakage, self-reference, and data integrity checks.

Tests run feature engineering but do not train any models (~1-2 min for the
session-scoped fixture, then each test is fast). Two main test classes cover
leakage detection and feature value validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import pearsonr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    import pandas as pd


# ---------------------------------------------------------------------------
# Leakage Tests
# ---------------------------------------------------------------------------

class TestLeakage:
    """Verify that no feature directly encodes the row's own target."""

    def test_d48_proxy_differs_from_own_demand(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand_d48_same_slot is not a perfect self-reference."""
        tr = pipeline_data["train_df"]
        d48 = tr[tr["day"] == 48]
        proxy = d48["demand_d48_same_slot"].values
        actual = d48["demand"].values
        pct_equal = np.mean(np.abs(proxy - actual) < 1e-10)
        assert pct_equal < 0.05, (
            f"{pct_equal:.1%} of day-48 rows have proxy == demand (leak threshold: 5%)"
        )

    def test_d48_proxy_correlation_below_unity(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand_d48_same_slot correlation with demand is below unity."""
        tr = pipeline_data["train_df"]
        d48 = tr[tr["day"] == 48]
        r, _ = pearsonr(d48["demand_d48_same_slot"].values, d48["demand"].values)
        assert r < 0.999, f"Proxy–demand Pearson r={r:.6f} is suspiciously close to 1"
        assert r > 0.5, f"Proxy–demand Pearson r={r:.6f} is too low — feature may be broken"

    def test_d49_loo_morning_mean_excludes_self(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand_d49_morning_mean excludes row's own demand (LOO property)."""
        tr = pipeline_data["train_df"]
        d49 = tr[tr["day"] == 49].copy()
        # Test only multi-row geohashes; single-row ones correctly use global mean.
        gh_counts = d49.groupby("geohash")["demand"].transform("count")
        multi = d49[gh_counts > 1]
        if multi.empty:
            pytest.skip("No multi-row geohashes on day-49")
        diff = np.abs(multi["demand_d49_morning_mean"].values - multi["demand"].values)
        pct_equal = np.mean(diff < 1e-10)
        assert pct_equal < 0.01, (
            f"{pct_equal:.1%} of day-49 multi-rows have morning_mean == demand (LOO broken)"
        )

    def test_d49_last_known_matches_test_regime(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand_d49_last_known is the carried-forward gh_last (2:00 value).

        The day-49 validation fold must see the SAME autoregressive anchor the real
        test rows get: the last known morning value per geohash (gh_last), carried
        forward. A fresh prior-slot anchor is unavailable on the test horizon
        (2:15–13:45) and would inflate the validation score, so within a geohash the
        feature must be constant across slots and equal to that geohash's 2:00 value.
        """
        tr = pipeline_data["train_df"]
        d49 = tr[tr["day"] == 49].sort_values(["geohash", "minute_of_day"])

        violations = 0
        checked = 0
        for gh, grp in d49.groupby("geohash"):
            demands = grp["demand"].tolist()
            last_knowns = grp["demand_d49_last_known"].tolist()
            gh_last = demands[-1]  # demand at the latest morning slot (≈2:00)
            for lk in last_knowns:
                checked += 1
                if abs(lk - gh_last) > 1e-10:
                    violations += 1

        if checked == 0:
            pytest.skip("No day-49 rows found")
        violation_rate = violations / checked
        assert violation_rate < 0.01, (
            f"{violation_rate:.1%} of day-49 rows have last_known != gh_last "
            f"(expected the carried-forward 2:00 value to match the test regime)"
        )

    def test_d48_rows_have_nonzero_d49_anchor(self, pipeline_data: dict[str, Any]) -> None:
        """Verify day-48 rows have non-zero day-49 morning anchor."""
        tr = pipeline_data["train_df"]
        d48 = tr[tr["day"] == 48]
        pct_nonzero = (d48["demand_d49_morning_mean"] > 0).mean()
        assert pct_nonzero > 0.95, (
            f"Only {pct_nonzero:.1%} of day-48 rows have non-zero demand_d49_morning_mean "
            f"(expected >95% — day-49 anchor fix may be missing)"
        )

    def test_geohash_te_fold_safe(self, pipeline_data: dict[str, Any]) -> None:
        """Verify target encoding excludes validation fold data (no leakage)."""
        from features import _smoothed_te
        from config import TE_M

        tr = pipeline_data["train_df"]
        y_log = pipeline_data["y_log"].values
        folds = pipeline_data["folds"]

        val_mask = folds == 0
        train_mask = folds == -1
        global_mean = float(y_log.mean())

        enc_from_train = _smoothed_te(
            tr["geohash"].values[train_mask],
            y_log[train_mask],
            TE_M,
            global_mean,
        )
        expected_te = (
            tr["geohash"].iloc[val_mask]
            .map(enc_from_train)
            .fillna(global_mean)
            .values
        )
        stored_te = tr["geohash_te"].values[val_mask]
        np.testing.assert_allclose(
            stored_te, expected_te, rtol=1e-5,
            err_msg="geohash_te for fold-0 rows does not match fold-safe recomputation"
        )

    def test_no_demand_column_in_feature_matrix(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand column is not present in feature matrix."""
        X_train = pipeline_data["X_train"]
        assert "demand" not in X_train.columns, (
            "'demand' column found in X_train — target is leaking into features"
        )


# ---------------------------------------------------------------------------
# Feature Integrity Tests
# ---------------------------------------------------------------------------

class TestFeatureIntegrity:
    """Verify feature values are finite, in-range, and correctly constructed."""

    def test_no_nan_in_x_train(self, pipeline_data: dict[str, Any]) -> None:
        """Verify X_train has no NaN values after imputation."""
        null_counts = pipeline_data["X_train"].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"NaN values found in X_train columns: {cols_with_nulls.to_dict()}"
        )

    def test_no_nan_in_x_test(self, pipeline_data: dict[str, Any]) -> None:
        """Verify X_test has no NaN values after imputation."""
        null_counts = pipeline_data["X_test"].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert cols_with_nulls.empty, (
            f"NaN values found in X_test columns: {cols_with_nulls.to_dict()}"
        )

    def test_cyclic_features_bounded(self, pipeline_data: dict[str, Any]) -> None:
        """Verify mod_sin and mod_cos are bounded in [-1, 1]."""
        X = pipeline_data["X_train"]
        for col in ("mod_sin", "mod_cos"):
            assert X[col].between(-1.0 - 1e-9, 1.0 + 1e-9).all(), (
                f"{col} has values outside [-1, 1]"
            )

    def test_lat_lon_plausible_range(self, pipeline_data: dict[str, Any]) -> None:
        """Verify lat/lon fall within valid geographic bounds."""
        X = pipeline_data["X_train"]
        assert X["lat"].between(-90, 90).all(), "lat out of [-90, 90]"
        assert X["lon"].between(-180, 180).all(), "lon out of [-180, 180]"

    def test_temperature_fully_imputed(self, pipeline_data: dict[str, Any]) -> None:
        """Verify Temperature has no missing values after imputation."""
        X_train = pipeline_data["X_train"]
        X_test = pipeline_data["X_test"]
        assert X_train["Temperature"].isnull().sum() == 0, "Missing Temperature in X_train"
        assert X_test["Temperature"].isnull().sum() == 0, "Missing Temperature in X_test"

    def test_minute_of_day_range(self, pipeline_data: dict[str, Any]) -> None:
        """Verify minute_of_day is in [0, 1425]."""
        X_train = pipeline_data["X_train"]
        X_test = pipeline_data["X_test"]
        for name, X in [("X_train", X_train), ("X_test", X_test)]:
            assert X["minute_of_day"].between(0, 1425).all(), (
                f"minute_of_day out of range in {name}"
            )

    def test_binary_features_are_zero_or_one(self, pipeline_data: dict[str, Any]) -> None:
        """Verify large_vehicles and landmarks are strictly binary."""
        X_train = pipeline_data["X_train"]
        for col in ("large_vehicles", "landmarks"):
            unique_vals = set(X_train[col].unique())
            assert unique_vals <= {0, 1}, (
                f"{col} has non-binary values: {unique_vals}"
            )

    def test_log_demand_d48_same_slot_is_log_of_same_slot(self, pipeline_data: dict[str, Any]) -> None:
        """Verify log_demand_d48_same_slot = log(demand_d48_same_slot)."""
        X = pipeline_data["X_train"]
        expected = np.log(np.clip(X["demand_d48_same_slot"].values, 1e-9, None))
        np.testing.assert_allclose(
            X["log_demand_d48_same_slot"].values,
            expected,
            rtol=1e-5,
            err_msg="log_demand_d48_same_slot is not log(demand_d48_same_slot)",
        )

    def test_demand_d48_relative_slot_formula(self, pipeline_data: dict[str, Any]) -> None:
        """Verify demand_d48_relative_slot = same_slot / geohash_mean."""
        X = pipeline_data["X_train"]
        EPS = 1e-9
        expected = X["demand_d48_same_slot"].values / (X["demand_d48_geohash_mean"].values + EPS)
        np.testing.assert_allclose(
            X["demand_d48_relative_slot"].values,
            expected,
            rtol=1e-5,
            err_msg="demand_d48_relative_slot formula mismatch",
        )

    def test_feature_count(self, pipeline_data: dict[str, Any]) -> None:
        """Verify feature matrix has the expected column count."""
        from config import CAT_FEATURES, NUMERIC_FEATURES
        expected = len(NUMERIC_FEATURES) + len(CAT_FEATURES)
        actual = pipeline_data["X_train"].shape[1]
        assert actual == expected, (
            f"Expected {expected} features, got {actual}"
        )

    def test_train_test_geohash_overlap(self, pipeline_data: dict[str, Any], raw_data: tuple) -> None:
        """Verify most test geohashes exist in training data."""
        train_df_raw, test_df_raw = raw_data
        train_ghs = set(train_df_raw["geohash"].unique())
        test_ghs = set(test_df_raw["geohash"].unique())
        cold_fraction = len(test_ghs - train_ghs) / len(test_ghs)
        assert cold_fraction < 0.10, (
            f"{cold_fraction:.1%} of test geohashes are unseen in training "
            f"(cold-start spatial fallback will be heavily used)"
        )

    def test_target_log_transform_invertible(self, pipeline_data: dict[str, Any]) -> None:
        """Verify exp(y_log) recovers original demand."""
        tr = pipeline_data["train_df"]
        y_log = pipeline_data["y_log"].values
        recovered = np.exp(y_log)
        original = tr["demand"].values
        np.testing.assert_allclose(
            recovered, original, rtol=1e-5,
            err_msg="exp(y_log) does not recover original demand — transform is incorrect",
        )
