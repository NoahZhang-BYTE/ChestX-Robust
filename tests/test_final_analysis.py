import numpy as np

from final_analysis import confusion_counts, top_error_rows


def test_confusion_counts_returns_tp_fp_tn_fn_per_label():
    targets = np.array([[1, 0], [0, 1], [1, 1]])
    probabilities = np.array([[0.8, 0.2], [0.7, 0.9], [0.1, 0.6]])
    result = confusion_counts(targets, probabilities, [0.5, 0.5])
    assert result == [
        {"tp": 1, "fp": 1, "tn": 0, "fn": 1},
        {"tp": 2, "fp": 0, "tn": 1, "fn": 0},
    ]


def test_top_error_rows_are_ranked_and_labelled():
    targets = np.array([[1, 0], [0, 1], [0, 0], [1, 0]])
    probabilities = np.array([[0.9, 0.8], [0.7, 0.6], [0.4, 0.95], [0.2, 0.1]])
    rows = top_error_rows(
        targets,
        probabilities,
        [0.5, 0.5],
        ["Pneumonia", "Fibrosis"],
        top_n=2,
        ids=["a", "b", "c", "d"],
    )
    assert rows[0]["error_type"] == "FP"
    assert rows[0]["label"] == "Fibrosis"
    assert rows[0]["sample_id"] == "c"
    assert rows[0]["score"] == 0.95
    assert rows[-1]["error_type"] == "FN"
