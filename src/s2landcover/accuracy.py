"""Accuracy assessment for a classified map against reference data.

Reports the full confusion matrix plus per-class producer's and user's
accuracy, not just overall accuracy and kappa.

Overall accuracy alone hides the failure that matters most in land cover work.
Classes are almost never balanced: a scene that is 60 percent open savanna
rewards a classifier for calling everything savanna, and the minority classes,
which are usually the ones the map was made for, can have near-zero recall
while OA still looks respectable. Per-class producer's accuracy is what exposes
that, so it is always reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["ConfusionResult", "confusion_matrix", "assess"]


@dataclass
class ConfusionResult:
    """Accuracy metrics for one classified map."""

    labels: List[int]
    matrix: np.ndarray                       # rows = reference, cols = predicted
    overall_accuracy: float
    kappa: float
    producers_accuracy: Dict[int, float] = field(default_factory=dict)
    users_accuracy: Dict[int, float] = field(default_factory=dict)
    n: int = 0

    def report(self, names: Dict[int, str] | None = None) -> str:
        names = names or {}
        lines = [
            "n = {:,} samples".format(self.n),
            "Overall accuracy : {:.1%}".format(self.overall_accuracy),
            "Cohen's kappa    : {:.3f}".format(self.kappa),
            "",
            "{:<18} {:>12} {:>12}".format("class", "producer's", "user's"),
        ]
        for lab in self.labels:
            lines.append("{:<18} {:>11.1%} {:>12.1%}".format(
                names.get(lab, str(lab)),
                self.producers_accuracy.get(lab, float("nan")),
                self.users_accuracy.get(lab, float("nan")),
            ))
        return "\n".join(lines)


def confusion_matrix(reference: np.ndarray, predicted: np.ndarray,
                     labels: Sequence[int] | None = None) -> tuple:
    """Confusion matrix with reference on rows and prediction on columns.

    Pixels where either array is 0 (no-data) are dropped, so the sample count
    reflects only pixels that were actually assessable.
    """
    reference = np.asarray(reference).ravel()
    predicted = np.asarray(predicted).ravel()
    if reference.shape != predicted.shape:
        raise ValueError("reference and predicted differ in size: {} vs {}".format(
            reference.size, predicted.size))

    keep = (reference != 0) & (predicted != 0)
    reference = reference[keep]
    predicted = predicted[keep]

    if labels is None:
        labels = sorted(set(np.unique(reference)) | set(np.unique(predicted)))
    labels = [int(x) for x in labels]
    index = {lab: i for i, lab in enumerate(labels)}

    m = np.zeros((len(labels), len(labels)), dtype="int64")
    for r, p in zip(reference, predicted):
        ri, pi = index.get(int(r)), index.get(int(p))
        if ri is not None and pi is not None:
            m[ri, pi] += 1
    return labels, m


def assess(reference: np.ndarray, predicted: np.ndarray,
           labels: Sequence[int] | None = None) -> ConfusionResult:
    """Full accuracy assessment.

    Producer's accuracy is recall: of the reference pixels in a class, how many
    were found. User's accuracy is precision: of the pixels mapped as a class,
    how many were right. Both are reported because they fail in opposite
    directions, and a burnt-area map can look good on one while being useless
    on the other.
    """
    labels, m = confusion_matrix(reference, predicted, labels)
    n = int(m.sum())
    if n == 0:
        raise ValueError("no assessable pixels: reference and predicted are "
                         "entirely no-data, or they do not overlap")

    correct = int(np.trace(m))
    oa = correct / n

    # Cohen's kappa: agreement corrected for what chance alone would give.
    row = m.sum(axis=1)
    col = m.sum(axis=0)
    expected = float((row * col).sum()) / (n * n)
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 1.0

    producers: Dict[int, float] = {}
    users: Dict[int, float] = {}
    for i, lab in enumerate(labels):
        producers[lab] = float(m[i, i] / row[i]) if row[i] else float("nan")
        users[lab] = float(m[i, i] / col[i]) if col[i] else float("nan")

    return ConfusionResult(
        labels=labels, matrix=m, overall_accuracy=oa, kappa=kappa,
        producers_accuracy=producers, users_accuracy=users, n=n,
    )
