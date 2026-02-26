import pandas as pd
import numpy as np

def main():
    print("Starting cleaning...")

    input_path = "data/processed/combined_flights.csv"
    output_path = "data/processed/ny_flights_clean.csv"

    # EXACT columns you used in your notebook
    use_cols = [
        "FlightDate", "Reporting_Airline",
        "Origin", "OriginCityName", "OriginState",
        "Dest", "DestCityName", "DestState",
        "CRSDepTime",
        "DepDelay", "DepDel15", "DepTimeBlk",
        "CRSArrTime", "ArrDelay",
        "ArrDel15",
        "Cancelled"
    ]

    # Read only the needed columns (saves a LOT of memory)
    df = pd.read_csv(input_path, usecols=use_cols)

    #  Keeping only NY origin flights
    df = df[df["OriginState"] == "NY"].copy()
    print("Rows after NY filter:", df.shape)

    #  Drop columns that are unnecessary 
    drop_cols = ["DepDelay", "DepDel15", "DepTimeBlk", "CRSArrTime", "ArrDelay", "Cancelled","OriginCityName", "OriginState", "DestCityName"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    #. Creating Date Features from FlightDate column
    df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    df["Day"] = df["FlightDate"].dt.day
    df["Month"] = df["FlightDate"].dt.month
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek + 1  # (optional) 1-7 like many datasets
    df["DayName"] = df["FlightDate"].dt.day_name()
    df["IsWeekend"] = df["DayName"].isin(["Saturday", "Sunday"]).astype(int)

    # 4)Creating isFixedHoliday
    fixed_holidays = {(1, 1), (7, 4), (11, 11), (12, 25)}
    df["IsFixedHoliday"] = df.apply(
        lambda r: int((r["Month"], r["Day"]) in fixed_holidays) if pd.notna(r["FlightDate"]) else 0,
        axis=1
    )

    # Creating isHolidayWindow feature for Holidays like thanksgiving and Christmas
    df["IsHolidayWindow"] = 0
    for (m, d) in fixed_holidays:
        mask = (df["Month"] == m) & (df["Day"].between(d - 1, d + 1))
        df.loc[mask, "IsHolidayWindow"] = 1

    # Creating DepHour column from CRSDepTime
    df["CRSDepTime"] = pd.to_numeric(df["CRSDepTime"], errors="coerce")
    df["DepHour"] = (df["CRSDepTime"] // 100).astype("Int64")

    #  Creating isRushHour feature
    df["IsRushHour"] = df["DepHour"].between(6, 10).astype("Int64")
    # Droping  IsRushHour 
    df.drop(columns=["IsRushHour"], inplace=True, errors="ignore")


    #  Drop missing target ArrDel15 (you did this)
    df = df[df["ArrDel15"].notna()].copy()

    # Save
    df.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print("Final shape:", df.shape)
    print("Target distribution:\n", df["ArrDel15"].value_counts(dropna=False))

if __name__ == "__main__":
    main()