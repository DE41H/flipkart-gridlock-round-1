"""Gridlock Hackathon 2.0 - demand forecasting pipeline orchestration.

Thin orchestrator that runs the modularized forecasting pipeline.
See config.py for constants and module docstrings for implementation details.
"""

from __future__ import annotations

import warnings

try:
    from .config import CONSOLE
    from .data import load_data
    from .preprocessing import parse_timestamps
    from .features import build_features
    from .model import train_models
    from .evaluation import evaluate
    from .reporting import blend_predictions, print_analytics, save_artifacts, save_run_log, save_submission
except ImportError:
    from config import CONSOLE
    from data import load_data
    from preprocessing import parse_timestamps
    from features import build_features
    from model import train_models
    from evaluation import evaluate
    from reporting import blend_predictions, print_analytics, save_artifacts, save_run_log, save_submission

warnings.filterwarnings("ignore")


def main() -> None:
    """Run the end-to-end forecasting pipeline."""
    CONSOLE.rule("[bold]Gridlock 2.0 - demand forecasting pipeline")
    train_df, test_df = load_data()

    test_parsed = parse_timestamps(test_df)

    X_train, y_log, X_test, feature_names, folds = build_features(train_df, test_df)

    res = train_models(X_train, y_log, X_test, folds)
    res = blend_predictions(res)
    eval_metrics = evaluate(res, X_train, X_test)
    save_artifacts(res, X_train, folds, eval_metrics=eval_metrics)
    print_analytics(res, eval_metrics)
    save_run_log(res, X_train, eval_metrics=eval_metrics)

    save_submission(test_parsed, res["test_pred"])

    CONSOLE.print(
        f"[bold]Final blended OOF R2: {res['blend_oof_r2']:.4f}[/]  "
        f"(LGBM {res['lgbm_oof_r2']:.4f} / CatBoost {res['cat_oof_r2']:.4f})"
    )


if __name__ == "__main__":
    main()
