"""Unit tests for the Phase-7 promotion gate — the guard-rail itself."""

from promotion import GateConfig, evaluate_gate

CHAMPION = {"pr_auc": 0.80, "cost_per_txn": 0.50}


def _challenger(pr_auc: float, cost: float) -> dict:
    return {"pr_auc": pr_auc, "cost_per_txn": cost}


def test_promotes_when_better_on_both():
    d = evaluate_gate(CHAMPION, _challenger(0.85, 0.45))
    assert d.promote
    assert d.comparison["pr_auc_delta"] > 0


def test_refuses_when_pr_auc_worse():
    d = evaluate_gate(CHAMPION, _challenger(0.75, 0.40))
    assert not d.promote
    assert any("PR-AUC did not improve" in r for r in d.reasons)


def test_refuses_when_cost_regresses_even_if_pr_auc_better():
    # Catches more fraud but flags so much that the cost per txn goes up:
    # exactly the model the gate exists to stop.
    d = evaluate_gate(CHAMPION, _challenger(0.90, 0.60))
    assert not d.promote
    assert any("cost/txn regressed" in r for r in d.reasons)


def test_tie_is_not_promoted():
    # Identical metrics must not churn the production model.
    d = evaluate_gate(CHAMPION, _challenger(0.80, 0.50))
    assert not d.promote


def test_min_gain_margin_is_respected():
    cfg = GateConfig(min_pr_auc_gain=0.01)
    assert not evaluate_gate(CHAMPION, _challenger(0.805, 0.45), cfg).promote
    assert evaluate_gate(CHAMPION, _challenger(0.82, 0.45), cfg).promote


def test_cost_tolerance_allows_bounded_regression():
    cfg = GateConfig(max_cost_regression=0.10)   # up to +10% cost is acceptable
    assert evaluate_gate(CHAMPION, _challenger(0.85, 0.54), cfg).promote
    assert not evaluate_gate(CHAMPION, _challenger(0.85, 0.60), cfg).promote


def test_bootstrap_promotes_without_champion():
    d = evaluate_gate(None, _challenger(0.70, 1.0))
    assert d.promote
    assert any("no production model" in r for r in d.reasons)


def test_absolute_floor_vetoes_even_bootstrap():
    # A broken run (near-random PR-AUC) is never promoted, champion or not.
    d = evaluate_gate(None, _challenger(0.05, 0.10))
    assert not d.promote
    assert any("absolute floor" in r for r in d.reasons)

    d2 = evaluate_gate({"pr_auc": 0.02, "cost_per_txn": 9.9}, _challenger(0.05, 0.10))
    assert not d2.promote


def test_decision_summary_is_readable():
    d = evaluate_gate(CHAMPION, _challenger(0.85, 0.45))
    text = d.summary()
    assert "PROMOTE" in text
    assert "PR-AUC" in text
