"""Autograder for Integration 5B — Model Comparison & Decision Memo.

Validates structural correctness and relative behavior. Assertions are
tied to ratios, baselines, and structural properties rather than
hardcoded numeric targets — matching realistic ML behavior and the
dataset's pedagogical properties.
"""

import ast
import inspect
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from joblib import load as joblib_load
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model_comparison import (
    NUMERIC_FEATURES,
    define_models,
    find_tree_vs_linear_disagreement,
    load_and_preprocess,
    log_experiment,
    plot_calibration_top3,
    plot_pr_curves_top3,
    run_cv_comparison,
    save_best_model,
    save_comparison_table,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def data():
    """Ensure cwd is repo root so load_and_preprocess finds data/telecom_churn.csv."""
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    result = load_and_preprocess()
    assert result is not None, "load_and_preprocess() returned None — implement Task 1"
    return result


@pytest.fixture(scope="module")
def models():
    """Define model configurations once."""
    result = define_models()
    assert result is not None, "define_models() returned None — implement Task 2"
    return result


@pytest.fixture(scope="module")
def cv_results(models, data):
    """Run CV once, reuse across tests. Module-scoped for speed."""
    X_train, X_test, y_train, y_test = data
    result = run_cv_comparison(models, X_train, y_train)
    assert result is not None, "run_cv_comparison() returned None — implement Task 3"
    return result


@pytest.fixture(scope="module")
def fitted_models(models, data):
    """Fit all models on training data once; reuse across tests."""
    X_train, X_test, y_train, y_test = data
    fitted = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted


# ─── Task 1: Data loading + split ────────────────────────────────────────────

def test_load_and_preprocess_shape(data):
    X_train, X_test, y_train, y_test = data
    total = len(X_train) + len(X_test)
    assert total >= 4000, f"Dataset too small: {total} rows"
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio:.3f} not ~0.20"


def test_load_and_preprocess_stratification(data):
    X_train, X_test, y_train, y_test = data
    train_rate = float(y_train.mean())
    test_rate = float(y_test.mean())
    assert abs(train_rate - test_rate) < 0.02, (
        f"Stratification not preserved: train={train_rate:.3f}, test={test_rate:.3f}. "
        "Use stratify=y in train_test_split."
    )


def test_load_and_preprocess_features(data):
    X_train, X_test, y_train, y_test = data
    assert list(X_train.columns) == NUMERIC_FEATURES, (
        f"X_train columns should be exactly NUMERIC_FEATURES. Got {list(X_train.columns)}"
    )


# ─── Task 2: Model definitions ───────────────────────────────────────────────

def test_define_models_has_6_configurations(models):
    assert len(models) == 6, (
        f"Expected exactly 6 model configurations, got {len(models)}"
    )


def test_define_models_are_pipelines(models):
    for name, pipe in models.items():
        assert isinstance(pipe, Pipeline), (
            f"'{name}' should be a sklearn Pipeline, got {type(pipe).__name__}"
        )


def test_define_models_includes_dummy(models):
    names_lower = [n.lower() for n in models.keys()]
    assert any("dummy" in n for n in names_lower), (
        "Models must include a DummyClassifier baseline"
    )


def test_define_models_includes_both_families(models):
    names_lower = [n.lower() for n in models.keys()]
    has_lr = any("lr" in n or "log" in n for n in names_lower)
    has_tree = any("rf" in n or "forest" in n or "tree" in n or "dt" in n
                   for n in names_lower)
    assert has_lr, "Models must include LogisticRegression variants"
    assert has_tree, "Models must include tree-based model variants"


def test_define_models_has_balanced_variants(models):
    """The 2x2 default-vs-balanced pattern must be present."""
    names_lower = [n.lower() for n in models.keys()]
    has_balanced = sum("balanced" in n for n in names_lower)
    assert has_balanced >= 2, (
        f"Expected at least 2 'balanced' model variants (LR + RF), found {has_balanced}"
    )


# ─── Task 3: Cross-validation comparison ─────────────────────────────────────

def test_cv_results_shape(cv_results):
    assert isinstance(cv_results, pd.DataFrame), "run_cv_comparison must return a DataFrame"
    assert len(cv_results) == 6, f"Expected 6 rows (one per model), got {len(cv_results)}"


def test_cv_results_has_required_columns(cv_results):
    required = ["model", "accuracy_mean", "accuracy_std", "precision_mean",
                "precision_std", "recall_mean", "recall_std", "f1_mean",
                "f1_std", "pr_auc_mean", "pr_auc_std"]
    missing = [c for c in required if c not in cv_results.columns]
    assert not missing, f"Missing columns in CV results: {missing}"


def test_cv_results_metrics_in_valid_range(cv_results):
    metric_cols = [c for c in cv_results.columns if c.endswith("_mean")]
    for col in metric_cols:
        values = cv_results[col].astype(float)
        assert all(0.0 <= v <= 1.0 for v in values), (
            f"All {col} values must be in [0, 1]. Got: {values.tolist()}"
        )


def test_dummy_pr_auc_below_baseline_gap(cv_results):
    """Dummy PR-AUC should be clearly separated from real models."""
    dummy_row = cv_results[cv_results["model"].str.lower().str.contains("dummy")]
    assert len(dummy_row) == 1, "Expected exactly one Dummy model row"
    dummy_auc = float(dummy_row["pr_auc_mean"].iloc[0])
    non_dummy = cv_results[~cv_results["model"].str.lower().str.contains("dummy")]
    worst_real = float(non_dummy["pr_auc_mean"].min())
    assert dummy_auc < worst_real, (
        f"Dummy PR-AUC ({dummy_auc:.3f}) should be strictly below the worst "
        f"real model ({worst_real:.3f})"
    )


def test_each_model_pr_auc_above_baseline(cv_results):
    """Every non-Dummy model should beat 1.5x the Dummy PR-AUC (positive rate)."""
    dummy_row = cv_results[cv_results["model"].str.lower().str.contains("dummy")]
    dummy_auc = float(dummy_row["pr_auc_mean"].iloc[0])
    non_dummy = cv_results[~cv_results["model"].str.lower().str.contains("dummy")]
    for _, row in non_dummy.iterrows():
        auc = float(row["pr_auc_mean"])
        assert auc >= 1.5 * dummy_auc, (
            f"{row['model']} PR-AUC ({auc:.3f}) should be >= 1.5x Dummy "
            f"({dummy_auc:.3f}). The model isn't learning."
        )


def test_class_weight_recall_shift_lr_level(cv_results):
    """LR balanced recall@0.5 should be materially higher than LR default.

    class_weight='balanced' shifts the operating point at the default 0.5
    threshold — this is the correct framing (not "improves recall").
    """
    lr_default = cv_results[cv_results["model"].str.lower().str.contains("lr") &
                            cv_results["model"].str.lower().str.contains("default")]
    lr_balanced = cv_results[cv_results["model"].str.lower().str.contains("lr") &
                             cv_results["model"].str.lower().str.contains("balanced")]
    if len(lr_default) == 1 and len(lr_balanced) == 1:
        r_def = float(lr_default["recall_mean"].iloc[0])
        r_bal = float(lr_balanced["recall_mean"].iloc[0])
        if r_def > 0:
            ratio = r_bal / r_def
            assert ratio >= 1.5, (
                f"LR balanced recall ({r_bal:.3f}) should be >= 1.5x LR default "
                f"recall ({r_def:.3f}) at the default 0.5 threshold. Got ratio={ratio:.2f}x"
            )


def test_class_weight_recall_shift_rf_level(cv_results):
    """RF balanced recall should be materially higher than RF default."""
    rf_default = cv_results[cv_results["model"].str.lower().str.contains("rf") &
                            cv_results["model"].str.lower().str.contains("default")]
    rf_balanced = cv_results[cv_results["model"].str.lower().str.contains("rf") &
                             cv_results["model"].str.lower().str.contains("balanced")]
    if len(rf_default) == 1 and len(rf_balanced) == 1:
        r_def = float(rf_default["recall_mean"].iloc[0])
        r_bal = float(rf_balanced["recall_mean"].iloc[0])
        if r_def > 0:
            ratio = r_bal / r_def
            assert ratio >= 1.5, (
                f"RF balanced recall ({r_bal:.3f}) should be >= 1.5x RF default "
                f"recall ({r_def:.3f}) at the default 0.5 threshold. Got ratio={ratio:.2f}x"
            )


# ─── Task 4: Comparison table saved ──────────────────────────────────────────

def test_comparison_table_saved_to_csv(tmp_path, cv_results):
    output = tmp_path / "comparison_table.csv"
    save_comparison_table(cv_results, str(output))
    assert output.exists(), "save_comparison_table did not create the file"
    saved = pd.read_csv(output)
    assert len(saved) >= 6, f"CSV should have >= 6 rows, got {len(saved)}"


# ─── Task 5: PR curves ───────────────────────────────────────────────────────

def test_pr_curves_plot_created(tmp_path, fitted_models, data):
    X_train, X_test, y_train, y_test = data
    output = tmp_path / "pr_curves.png"
    plot_pr_curves_top3(fitted_models, X_test, y_test, str(output))
    assert output.exists(), "plot_pr_curves_top3 did not save a file"
    assert output.stat().st_size > 1000, (
        "PR curve PNG is suspiciously small — did you call plt.savefig AFTER plotting?"
    )


# ─── Task 6: Calibration plot ────────────────────────────────────────────────

def test_calibration_plot_created(tmp_path, fitted_models, data):
    X_train, X_test, y_train, y_test = data
    output = tmp_path / "calibration.png"
    plot_calibration_top3(fitted_models, X_test, y_test, str(output))
    assert output.exists(), "plot_calibration_top3 did not save a file"
    assert output.stat().st_size > 1000, (
        "Calibration PNG is suspiciously small — did you call plt.savefig AFTER plotting?"
    )


# ─── Task 7: Best model saved ────────────────────────────────────────────────

def test_best_model_saved_as_joblib(tmp_path, fitted_models, cv_results):
    best_name = cv_results.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    output = tmp_path / "best_model.joblib"
    save_best_model(fitted_models[best_name], str(output))
    assert output.exists(), "save_best_model did not create the file"
    loaded = joblib_load(str(output))
    assert hasattr(loaded, "predict_proba"), (
        "Loaded model must have predict_proba (it's a Pipeline with a classifier)"
    )


# ─── Task 8: Experiment log ──────────────────────────────────────────────────

def test_experiment_log_has_all_models(tmp_path, cv_results):
    output = tmp_path / "experiment_log.csv"
    log_experiment(cv_results, str(output))
    assert output.exists(), "log_experiment did not create the file"
    log_df = pd.read_csv(output)
    assert len(log_df) >= 6, f"Experiment log should have >= 6 rows, got {len(log_df)}"
    required_cols = {"model_name", "accuracy", "precision", "recall", "f1",
                     "pr_auc", "timestamp"}
    missing = required_cols - set(log_df.columns)
    assert not missing, f"Experiment log missing columns: {missing}"


# ─── Task 9: Tree-vs-linear disagreement ─────────────────────────────────────

def test_tree_vs_linear_disagreement_structure(fitted_models, data):
    """Disagreement finding must return a valid dict with meaningful diff."""
    X_train, X_test, y_train, y_test = data
    # Get RF and LR pipelines
    rf_key = [k for k in fitted_models if "rf" in k.lower() and "balanced" not in k.lower()]
    lr_key = [k for k in fitted_models if "lr" in k.lower() and "balanced" not in k.lower()]
    assert rf_key, "No RF_default model found in fitted_models"
    assert lr_key, "No LR_default model found in fitted_models"

    d = find_tree_vs_linear_disagreement(
        fitted_models[rf_key[0]], fitted_models[lr_key[0]],
        X_test, y_test, NUMERIC_FEATURES
    )
    assert d is not None, "find_tree_vs_linear_disagreement() returned None"
    required = {"sample_idx", "feature_values", "rf_proba", "lr_proba",
                "prob_diff", "true_label"}
    assert required.issubset(d.keys()), (
        f"Missing keys in disagreement dict: {required - set(d.keys())}"
    )
    assert isinstance(d["sample_idx"], (int, np.integer))
    assert isinstance(d["feature_values"], dict)
    assert len(d["feature_values"]) == len(NUMERIC_FEATURES), (
        f"feature_values should have {len(NUMERIC_FEATURES)} keys"
    )
    assert 0.0 <= float(d["rf_proba"]) <= 1.0
    assert 0.0 <= float(d["lr_proba"]) <= 1.0
    assert float(d["prob_diff"]) >= 0.15, (
        f"prob_diff should be >= 0.15 (meaningful disagreement); got {d['prob_diff']:.3f}"
    )
    assert int(d["true_label"]) in (0, 1)
    # Consistency: diff should equal |rf_proba - lr_proba|
    computed_diff = abs(float(d["rf_proba"]) - float(d["lr_proba"]))
    assert abs(computed_diff - float(d["prob_diff"])) < 1e-6


def test_tree_vs_linear_disagreement_uses_predict_proba():
    """find_tree_vs_linear_disagreement must use predict_proba, not predict."""
    source = inspect.getsource(find_tree_vs_linear_disagreement)
    assert "predict_proba" in source, (
        "find_tree_vs_linear_disagreement must call predict_proba to compare "
        "probability estimates between models."
    )


# ─── CV uses StratifiedKFold (AST check) ─────────────────────────────────────

def test_cv_comparison_uses_stratified_kfold():
    """run_cv_comparison should use StratifiedKFold for proper evaluation."""
    source = inspect.getsource(run_cv_comparison)
    assert "StratifiedKFold" in source, (
        "run_cv_comparison must use StratifiedKFold for cross-validation "
        "to maintain class balance across folds."
    )
