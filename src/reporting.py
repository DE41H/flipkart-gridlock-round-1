"""Analytics and submission reporting module for Gridlock 2.0 pipeline."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rich.table import Table

try:
    from .config import CLIP_HI, CLIP_LO, CONSOLE, SUB_DIR, SUB_PATH
    from .model import _r2_original_scale
except ImportError:
    from config import CLIP_HI, CLIP_LO, CONSOLE, SUB_DIR, SUB_PATH
    from model import _r2_original_scale


def blend_predictions(res: dict) -> dict:
    """Find optimal LGBM/CatBoost blend weight and apply to predictions.

    Optimizes LGBM weight w (formula: w*LGBM + (1-w)*CatBoost) to maximize OOF R2
    on original demand scale. Applies blend to test predictions, inverse-transforms,
    and clips.

    Parameters
    ----------
    res : dict
        Dictionary from train_models() containing lgbm_oof, cat_oof, lgbm_test,
        cat_test, y_log, folds, and other metadata.

    Returns
    -------
    dict
        Updated res with blend metrics:
          - lgbm_oof_r2, cat_oof_r2: individual OOF R2
          - blend_oof_r2: blended OOF R2
          - blend_weight: optimal LGBM weight
          - test_pred: final clipped test predictions
    """
    CONSOLE.print("[bold cyan]Blending predictions...[/]")
    folds = res["folds"]
    oof_mask = folds >= 0
    y_log = res["y_log"][oof_mask]
    lgbm_oof = res["lgbm_oof"][oof_mask]
    cat_oof = res["cat_oof"][oof_mask]

    lgbm_r2 = _r2_original_scale(y_log, lgbm_oof)
    cat_r2 = _r2_original_scale(y_log, cat_oof)

    best_w, best_r2 = 0.5, -np.inf
    for w in np.linspace(0.0, 1.0, 51):
        r2 = _r2_original_scale(y_log, w * lgbm_oof + (1 - w) * cat_oof)
        if r2 > best_r2:
            best_r2, best_w = r2, w

    blend_test_log = best_w * res["lgbm_test"] + (1 - best_w) * res["cat_test"]
    test_pred = np.clip(np.exp(blend_test_log), CLIP_LO, CLIP_HI)

    res.update(
        {
            "lgbm_oof_r2": lgbm_r2,
            "cat_oof_r2": cat_r2,
            "blend_oof_r2": best_r2,
            "blend_weight": best_w,
            "test_pred": test_pred,
        }
    )
    CONSOLE.print(
        f"  LGBM OOF R2={lgbm_r2:.4f}  CatBoost OOF R2={cat_r2:.4f}  "
        f"blend w(LGBM)={best_w:.2f} -> OOF R2={best_r2:.4f}"
    )
    return res


def print_analytics(res: dict) -> None:
    """Print rich console tables with CV metrics, blend results, and feature importances.

    Parameters
    ----------
    res : dict
        Results dictionary from train_models() and blend_predictions() containing
        fold_rows, OOF/blend metrics, importances, and test predictions.
    """
    t = Table(title="Temporal CV - per fold")
    for col in ["Fold", "Train R2", "Val R2", "LGBM best_iter", "CatBoost best_iter"]:
        t.add_column(col, justify="right")
    for r in res["fold_rows"]:
        t.add_row(
            str(r["fold"]),
            f"{r['train_r2']:.4f}",
            f"{r['val_r2']:.4f}",
            str(r["lgbm_best_iter"]),
            str(r["cat_best_iter"]),
        )
    CONSOLE.print(t)

    t2 = Table(title="OOF performance & blend")
    for col in ["Model", "OOF R2", "Blend weight"]:
        t2.add_column(col, justify="right")
    t2.add_row("LightGBM", f"{res['lgbm_oof_r2']:.4f}", f"{res['blend_weight']:.2f}")
    t2.add_row("CatBoost", f"{res['cat_oof_r2']:.4f}", f"{1 - res['blend_weight']:.2f}")
    t2.add_row("Blend", f"{res['blend_oof_r2']:.4f}", "-")
    CONSOLE.print(t2)

    t3 = Table(title="LightGBM feature importance (gain, top 20)")
    t3.add_column("Feature")
    t3.add_column("Importance", justify="right")
    for feat, imp in res["importances"].head(20).items():
        t3.add_row(str(feat), f"{imp:,.0f}")
    CONSOLE.print(t3)

    pred = res["test_pred"]
    pct_clipped = float(np.mean((pred <= CLIP_LO + 1e-12) | (pred >= CLIP_HI - 1e-12)) * 100)
    t4 = Table(title="Test prediction distribution")
    t4.add_column("Stat")
    t4.add_column("Value", justify="right")
    t4.add_row("min", f"{pred.min():.6f}")
    t4.add_row("max", f"{pred.max():.6f}")
    t4.add_row("mean", f"{pred.mean():.6f}")
    t4.add_row("median", f"{np.median(pred):.6f}")
    t4.add_row("std", f"{pred.std():.6f}")
    t4.add_row("% clipped", f"{pct_clipped:.2f}%")
    CONSOLE.print(t4)


def save_submission(test_df: pd.DataFrame, test_pred: np.ndarray) -> None:
    """Write submission CSV with Index and demand columns.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test DataFrame with 'Index' column.
    test_pred : np.ndarray
        Predicted demand values.
    """
    os.makedirs(SUB_DIR, exist_ok=True)
    sub = pd.DataFrame({"Index": test_df["Index"].to_numpy(), "demand": test_pred})
    sub.to_csv(SUB_PATH, index=False)
    CONSOLE.print(
        f"[bold green]Submission saved -> submissions/submission.csv  |  "
        f"Shape: {sub.shape[0]} x {sub.shape[1]}[/]"
    )
