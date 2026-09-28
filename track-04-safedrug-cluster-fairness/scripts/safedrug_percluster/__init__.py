"""SafeDrug per-cluster evaluation helpers.

See docs/superpowers/specs/2026-09-04-safedrug-per-cluster-eval-design.md.
"""

from __future__ import annotations

from .metrics import attach_labels, patient_bootstrap_ci, visit_metrics

__all__ = ["visit_metrics", "attach_labels", "patient_bootstrap_ci"]
