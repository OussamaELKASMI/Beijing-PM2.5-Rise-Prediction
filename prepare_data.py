import csv
from datetime import datetime, timedelta
from pathlib import Path

input_path = (
    Path("PRSA_Data_20130301-20170228")
    / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)
output_path = Path("pm25_labeled.csv")


def read_timestamp(row):
    # Combine the year, month, day, and hour columns into one timestamp.
    return datetime(
        int(row["year"]), int(row["month"]), int(row["day"]), int(row["hour"])
    )


with input_path.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))  # Read all station rows in file order.

saved = 0
skipped_gaps = 0
skipped_missing = 0

with output_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "pm25", "temperature", "wind_speed", "label"])

    # Pair row 1 with row 2, row 2 with row 3, and so on.
    for current, next_row in zip(rows, rows[1:]):
        timestamp = read_timestamp(current)

        if read_timestamp(next_row) - timestamp != timedelta(hours=1):
            skipped_gaps += 1
            continue  # The label needs a reading exactly one hour later.

        # Take three inputs from the current hour and PM2.5 from the next hour.
        values = (
            current["PM2.5"], current["TEMP"], current["WSPM"], next_row["PM2.5"]
        )
        if any(value in ("NA", "") for value in values):
            skipped_missing += 1
            continue  # Skip this pair if a required reading is missing.

        pm25, temperature, wind_speed, next_pm25 = map(float, values)
        label = int(next_pm25 > pm25)  # 1 if PM2.5 rose, 0 if it stayed or fell.

        writer.writerow([
            timestamp.isoformat(" ", timespec="minutes"),
            pm25, temperature, wind_speed, label,
        ])
        saved += 1

print("CSV rows read:", len(rows))
print("Examples saved:", saved)
print("Pairs skipped for a time gap:", skipped_gaps)
print("Pairs skipped for missing values:", skipped_missing)