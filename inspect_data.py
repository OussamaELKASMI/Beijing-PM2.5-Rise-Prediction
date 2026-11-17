import csv

with open("PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv", encoding="utf-8-sig", newline="") as file:
    # rows = csv.reader(file)
    # print("Columns:", next(rows))
    
    # for _ in range(3):
    # print(next(rows))

    rows = csv.DictReader(file)
    current = next(rows)
    next_hour = next(rows)

    pm25 = float(current["PM2.5"])
    temperature = float(current["TEMP"])
    wind_speed = float(current["WSPM"])

    label = int(float(next_hour["PM2.5"]) > pm25)

    print("Inputs:", pm25, temperature, wind_speed)
    print("Label:", label)