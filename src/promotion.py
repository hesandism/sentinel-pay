"""
SentinelPay — model promotion gate (Phase 7)
============================================

The guard-rail that decides whether a freshly retrained model (the
**challenger**) may replace the currently serving model (the **champion**).
Deliberately a pure module: it takes two metric dicts and a config, returns a
decision with human-readable reasons, and touches nothing else — so the rule
is unit-testable without MLflow, Postgres, or a trained model anywhere near it.

The rule
--------
The challenger is promoted only if, on the SAME held-out evaluation window:

    1. its PR-AUC is at least ``min_pr_auc_gain`` better than the champion's
       (PR-AUC is the headline metric for ~0.5% fraud — accuracy is useless);
    2. its cost per transaction is no worse than the champion's (within
       ``max_cost_regression`` relative tolerance) — a model that catches more
       fraud by flagging everything must not sneak through; and
    3. its PR-AUC clears an absolute sanity floor, so a broken run can never
       be promoted just because the champion happens to be worse.

If there is no champion at all (empty registry), the challenger passes by
default — someone has to be first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GateConfig:
    """Tunable knobs of the promotion rule (defaults are deliberately strict)."""

    # Challenger PR-AUC must beat champion PR-AUC by at least this much.
    # A small positive margin means "a tie is not a reason to churn models".
    min_pr_auc_gain: float = 0.001

    # Challenger cost/txn may exceed champion cost/txn by at most this relative
    # fraction (0.0 = must be equal or cheaper).
    max_cost_regression: float = 0.0

    # Absolute floor: never promote a model whose PR-AUC is below this, no
    # matter how bad the champion is.
    min_pr_auc_floor: float = 0.10


@dataclass
class GateDecision:
    """The outcome of the gate, with the full paper trail."""

    promote: bool
    reasons: List[str] = field(default_factory=list)   # why it passed / failed
    comparison: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        verdict = "PROMOTE" if self.promote else "KEEP CHAMPION"
        lines = [f"Gate decision: {verdict}"] + [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def evaluate_gate(
    champion: Optional[Dict[str, float]],
    challenger: Dict[str, float],
    config: GateConfig = GateConfig(),
) -> GateDecision:
    """Apply the promotion rule.

    Parameters
    ----------
    champion : metric dict for the current Production model on the shared
        evaluation window (needs ``pr_auc`` and ``cost_per_txn``), or ``None``
        if the registry has no production model yet.
    challenger : same metric dict for the newly trained model.
    config : the gate thresholds.
    """
    ch_pr = float(challenger["pr_auc"])
    ch_cost = float(challenger["cost_per_txn"])

    # Sanity floor first: this can veto even a bootstrap promotion.
    if ch_pr < config.min_pr_auc_floor:
        return GateDecision(
            promote=False,
            reasons=[
                f"challenger PR-AUC {ch_pr:.4f} is below the absolute floor "
                f"{config.min_pr_auc_floor:.2f} — the run looks broken, refusing "
                "to promote regardless of the champion"
            ],
            comparison={"challenger_pr_auc": ch_pr, "challenger_cost_per_txn": ch_cost},
        )

    # Bootstrap: no champion to beat.
    if champion is None:
        return GateDecision(
            promote=True,
            reasons=["no production model exists yet — challenger promoted as the first champion"],
            comparison={"challenger_pr_auc": ch_pr, "challenger_cost_per_txn": ch_cost},
        )

    cm_pr = float(champion["pr_auc"])
    cm_cost = float(champion["cost_per_txn"])
    comparison = {
        "champion_pr_auc": cm_pr,
        "challenger_pr_auc": ch_pr,
        "pr_auc_delta": ch_pr - cm_pr,
        "champion_cost_per_txn": cm_cost,
        "challenger_cost_per_txn": ch_cost,
        "cost_per_txn_delta": ch_cost - cm_cost,
    }

    reasons: List[str] = []
    pr_ok = ch_pr >= cm_pr + config.min_pr_auc_gain
    if pr_ok:
        reasons.append(
            f"PR-AUC improved {cm_pr:.4f} -> {ch_pr:.4f} "
            f"(+{ch_pr - cm_pr:.4f}, required gain {config.min_pr_auc_gain})"
        )
    else:
        reasons.append(
            f"PR-AUC did not improve enough: {cm_pr:.4f} -> {ch_pr:.4f} "
            f"({ch_pr - cm_pr:+.4f}, required gain {config.min_pr_auc_gain})"
        )

    cost_ceiling = cm_cost * (1.0 + config.max_cost_regression)
    cost_ok = ch_cost <= cost_ceiling
    if cost_ok:
        reasons.append(
            f"cost/txn acceptable: ${cm_cost:.4f} -> ${ch_cost:.4f} "
            f"(ceiling ${cost_ceiling:.4f})"
        )
    else:
        reasons.append(
            f"cost/txn regressed: ${cm_cost:.4f} -> ${ch_cost:.4f} "
            f"(above the ceiling ${cost_ceiling:.4f})"
        )

    return GateDecision(promote=pr_ok and cost_ok, reasons=reasons, comparison=comparison)
