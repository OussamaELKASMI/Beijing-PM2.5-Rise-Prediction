"""Evaluation and terminal summaries for the PM2.5 experiment."""

import csv
from datetime import datetime, timedelta
from statistics import median

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_text


def majority_label(training_labels):
    """Return the most common training label, using 0 for ties."""
    rises = sum(training_labels)  # Only a next-hour rise has label 1.
    no_rises = len(training_labels) - rises
    return int(rises > no_rises)


def print_overall_results(tree, feature_names, example, actual_labels, predictions,
                          baseline_label):
    """Compare full test-set accuracy and show the fitted tree and its errors."""
    baseline_correct = sum(label == baseline_label for label in actual_labels)
    baseline_accuracy = baseline_correct / len(actual_labels)
    tree_accuracy = accuracy_score(actual_labels, predictions)
    difference = 100 * (tree_accuracy - baseline_accuracy)  # Percentage points.

    print("Baseline label:", baseline_label)
    print(f"Baseline test accuracy: {baseline_accuracy:.2%}")
    print(f"Decision tree test accuracy: {tree_accuracy:.2%}")
    print(f"Difference from baseline: {difference:+.2f} percentage points")
    print(export_text(tree, feature_names=feature_names))

    # Rows are actual labels and columns are predicted labels, ordered 0 then 1.
    print(confusion_matrix(actual_labels, predictions, labels=[0, 1]))
    print("Test example:", example)
    print("Tree prediction:", predictions[0])


def print_exploratory_summary(train_rows, test_rows, feature_names):
    """Show class counts for both splits and feature ranges for training."""
    print("\nEXPLORATORY DATA ANALYSIS")

    for name, rows in [("Training", train_rows), ("Test", test_rows)]:
        rises = sum(int(row["label"]) for row in rows)
        no_rises = len(rows) - rises
        print(f"{name}: {rises:,} rises, {no_rises:,} no-rise cases")
        print(f"{name} rise rate: {rises / len(rows):.2%}")

    for feature in feature_names:
        values = [float(row[feature]) for row in train_rows]
        print(
            f"{feature}: min={min(values):g}, "
            f"median={median(values):g}, max={max(values):g}"
        )


def print_seasonal_results(test_rows, predictions):
    """Compare the tree with the always-rise rule for each season."""
    seasons = {
        "Spring": (3, 4, 5),
        "Summer": (6, 7, 8),
        "Autumn": (9, 10, 11),
        "Winter": (12, 1, 2),
    }

    test_cases = []
    for row, prediction in zip(test_rows, predictions, strict=True):
        month = datetime.fromisoformat(row["timestamp"]).month  # Input month.
        test_cases.append((month, int(row["label"]), int(prediction)))

    print("\nSEASONAL TEST RESULTS")
    print(
        f"{'Season':<8} {'Test cases':>10} {'Baseline':>10} "
        f"{'Tree':>10} {'Gain (pp)':>10}"
    )

    for season, months in seasons.items():
        cases = [
            (actual, prediction)
            for month, actual, prediction in test_cases
            if month in months
        ]
        if not cases:
            continue

        # Class 1 represents a rise and was the majority label in the study.
        baseline_correct = sum(actual == 1 for actual, _ in cases)
        tree_correct = sum(actual == prediction for actual, prediction in cases)
        total = len(cases)
        gain = 100 * (tree_correct - baseline_correct) / total

        print(
            f"{season:<8} {total:>10,} {baseline_correct / total:>10.2%} "
            f"{tree_correct / total:>10.2%} {gain:>+10.2f}"
        )


def load_pm25_by_hour(station_csv):
    """Index the valid raw PM2.5 readings by timestamp."""
    pm25_by_hour = {}

    # A previous-hour reading can be useful even without a labeled example.
    with open(station_csv, encoding="utf-8-sig", newline="") as station_file:
        for row in csv.DictReader(station_file):
            value = row["PM2.5"]
            if value in ("NA", ""):
                continue

            hour = datetime(
                int(row["year"]),
                int(row["month"]),
                int(row["day"]),
                int(row["hour"]),
            )
            pm25_by_hour[hour] = float(value)

    return pm25_by_hour

def print_previous_hour_comparison(station_csv, test_rows, actual_labels,
                                   predictions, baseline_label):
    """Score all three methods on hours with a valid previous reading."""
    pm25_by_hour = load_pm25_by_hour(station_csv)
    matched = 0
    majority_correct = 0
    previous_hour_correct = 0
    tree_correct = 0

    for row, actual, tree_prediction in zip(
        test_rows, actual_labels, predictions, strict=True
    ):
        hour = datetime.fromisoformat(row["timestamp"])
        previous_pm25 = pm25_by_hour.get(hour - timedelta(hours=1))

        # Skip the hour for every method so all three scores use the same cases.
        if previous_pm25 is None:
            continue

        # Predict a rise if PM2.5 rose from the previous hour to the current hour.
        previous_hour_prediction = int(float(row["pm25"]) > previous_pm25)

        matched += 1
        majority_correct += int(baseline_label == actual)
        previous_hour_correct += int(previous_hour_prediction == actual)
        tree_correct += int(tree_prediction == actual)

    if matched == 0:
        raise SystemExit("No test hours had a valid previous PM2.5 reading.")

    print("\nPREVIOUS-HOUR BASELINE COMPARISON")
    print(f"Matched test hours: {matched:,} of {len(test_rows):,}")
    print(f"Excluded for missing previous PM2.5: {len(test_rows) - matched:,}")

    for method, correct in [
        ("Majority-class baseline", majority_correct),
        ("Previous-hour baseline", previous_hour_correct),
        ("Decision tree", tree_correct),
    ]:
        print(f"{method}: {correct:,}/{matched:,} ({correct / matched:.2%})")

    gain = 100 * (tree_correct - previous_hour_correct) / matched
    print(f"Tree minus previous-hour baseline: {gain:+.2f} percentage points")