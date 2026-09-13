import os

from ml_utils import classify_category, compute_urgency, compute_priority, find_duplicate


def test_classify_category_returns_known_label():
    label = classify_category("wifi signal is weak in the library")
    assert label in {"wifi", "other"}


def test_compute_urgency_ranges():
    score = compute_urgency("fire alarm is sparking and unsafe near the gate")
    assert 0.0 <= score <= 1.0
    assert score > 0.5


def test_compute_priority_positive():
    score = compute_priority(0.8, 2, 4)
    assert score > 0


def test_find_duplicate_uses_similarity():
    open_complaints = [
        (1, "wifi connection is down in Block A", "wifi", "Block A"),
        (2, "water leak in the hostel bathroom", "plumbing", "Hostel 2"),
    ]
    dup = find_duplicate("wifi connection is down again in Block A", "wifi", "Block A", open_complaints)
    assert dup == 1


def test_ragging_and_safety_classification():
    from ml_utils import infer_department
    text = "there are 5 boys who are beating up a single boy in the boys washroom in hostel2"
    assert classify_category(text) == "safety"
    assert compute_urgency(text) == 1.0
    assert infer_department("safety", text) == "Safety Team"

