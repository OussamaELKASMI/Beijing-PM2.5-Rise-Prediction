"""Train and evaluate a decision tree for next-hour PM2.5 increases."""

import csv
from datetime import datetime, timedelta

from sklearn.tree import DecisionTreeClassifier

from evaluation import (
    majority_label,
    print_exploratory_summary,
    print_overall_results,
    print_previous_hour_comparison,
    print_seasonal_results,
)

LABELED_CSV = "pm25_labeled.csv"
STATION_CSV = (
    "PRSA_Data_20130301-20170228/"
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)
CUTOFF = datetime(2016, 3, 1)  # First input hour in the test period.
FEATURE_NAMES = ["pm25", "temperature", "wind_speed"]


def split_rows(labeled_csv, cutoff):
    """Split by time and exclude examples whose label crosses the cutoff."""
    train_rows = []
    test_rows = []
    skipped_boundary = 0

    with open(labeled_csv, encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            timestamp = datetime.fromisoformat(row["timestamp"])
            # The timestamp is the input hour. The label uses the next hour.
            if timestamp >= cutoff:
                test_rows.append(row)
            elif timestamp + timedelta(hours=1) < cutoff:
                train_rows.append(row)  # Both hours precede the cutoff.
            else:
                skipped_boundary += 1  # This label uses a test-period reading.

    if not train_rows or not test_rows:
        raise SystemExit("Expected examples on both sides of the cutoff.")

    return train_rows, test_rows, skipped_boundary


def get_features(rows):
    """Read the three input features in the order the tree expects."""
    return [[float(row[name]) for name in FEATURE_NAMES] for row in rows]


def main():
    train_rows, test_rows, skipped_boundary = split_rows(LABELED_CSV, CUTOFF)
    X_train = get_features(train_rows)
    X_test = get_features(test_rows)
    y_train = [int(row["label"]) for row in train_rows]
    y_test = [int(row["label"]) for row in test_rows]  # Used only for scoring.
    baseline_label = majority_label(y_train)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)

    print("Training examples:", len(train_rows))
    print("Test examples:", len(test_rows))
    print("Boundary examples skipped:", skipped_boundary)

    print_overall_results(
        tree, FEATURE_NAMES, test_rows[0], y_test, predictions, baseline_label
    )
    print_exploratory_summary(train_rows, test_rows, FEATURE_NAMES)
    print_seasonal_results(test_rows, predictions)
    print_previous_hour_comparison(
        STATION_CSV, test_rows, y_test, predictions, baseline_label
    )


if __name__ == "__main__":
    main()