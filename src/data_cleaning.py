import pandas as pd
import numpy as np
import requests 

def main():
    print("Starting cleaning...")

    input_path = "data/processed/combined_flights.csv"
    output_path = "data/processed/ny_flights_clean.csv"

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


    #  Drop missing target ArrDel15 
    df = df[df["ArrDel15"].notna()].copy()

    #Creating Departure Time Category
    df["DepTimeCategory"] = np.where(
    df["DepHour"].between(5, 11), "Morning",
    np.where(
        df["DepHour"].between(12, 16), "Afternoon",
        np.where(
            df["DepHour"].between(17, 21), "Evening",
            "Late Night")))
    
    #Building Weather columns
    df["CRSDepTime_str"] = df["CRSDepTime"].fillna(0).astype(int).astype(str).str.zfill(4)

    df["dep_hour"] = df["CRSDepTime_str"].str[:2].astype(int)
    df["dep_min"] = df["CRSDepTime_str"].str[2:].astype(int)

    df["dep_local_dt"] = (
        df["FlightDate"]
        + pd.to_timedelta(df["dep_hour"], unit="h")
        + pd.to_timedelta(df["dep_min"], unit="m")
    )

    df["dep_utc_dt"] = (
        df["dep_local_dt"]
        .dt.tz_localize("America/New_York", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    df["dep_utc_hour"] = df["dep_utc_dt"].dt.floor("h")
    AIRPORT_COORDS = {
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "BUF": (42.9405, -78.7322),
    "ALB": (42.7483, -73.8017),
    "HPN": (41.0670, -73.7076),
    "SYR": (43.1112, -76.1063),
    "ROC": (43.1189, -77.6724),
    "ISP": (40.7952, -73.1002),
    "ELM": (42.1599, -76.8916),
    "SWF": (41.5041, -74.1048),
    "PBG": (44.6509, -73.4681),
    "IAG": (43.1073, -78.9462),
    "BGM": (42.2087, -75.9797),
    "ITH": (42.4910, -76.4584),}

    print("Fetching weather data...")

    start_date = df["FlightDate"].min().strftime("%Y-%m-%d")
    end_date = df["FlightDate"].max().strftime("%Y-%m-%d")

    weather_frames = []

    for airport, (lat, lon) in AIRPORT_COORDS.items():
        print(f"Getting weather for {airport}")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_direction_10m,wind_gusts_10m,visibility,cloud_cover",
            "timezone": "UTC"}

        r = requests.get(url, params=params, timeout=60)
        data = r.json()

        if "hourly" not in data:
            continue

        wx = pd.DataFrame({
        "dep_utc_hour": pd.to_datetime(data["hourly"]["time"], utc=True),
        "temp": data["hourly"]["temperature_2m"],
        "prcp": data["hourly"]["precipitation"],
        "rain": data["hourly"]["rain"],
        "snow": data["hourly"]["snowfall"],
        "wspd": data["hourly"]["wind_speed_10m"],
        "wdir": data["hourly"]["wind_direction_10m"],
        "gust": data["hourly"]["wind_gusts_10m"],
        "vis": data["hourly"]["visibility"],
        "cloud": data["hourly"]["cloud_cover"]})

        wx["Origin"] = airport
        weather_frames.append(wx)

    weather_df = pd.concat(weather_frames, ignore_index=True)
    print("Merging weather with flights...")

    df = df.merge(
        weather_df,
        on=["Origin", "dep_utc_hour"],
        how="left"
    )
    for c in ["rain", "snow", "wspd"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Rain_type
    df["Rain_type"] = np.where(
        df["rain"] > 6, "Heavy",
        np.where(
            df["rain"] > 2, "Moderate",
            np.where(df["rain"] > 0, "Normal", "None")
        )
    )

    # Snow_type
    df["Snow_type"] = np.where(
        df["snow"] > 5, "Heavy",
        np.where(
            df["snow"] > 2, "Moderate",
            np.where(df["snow"] > 0, "Normal", "None")
        )
    )

    # Wind_type
    df["Wind_type"] = np.where(
        df["wspd"] > 35, "Very_Windy",
        np.where(df["wspd"] >= 20, "Windy", "Normal")

    )
    df = df[df["temp"].notna()].copy()

    extra_drop_cols = ["DestState", "wdir", "gust", "cloud"]
    df.drop(columns=extra_drop_cols, inplace=True, errors="ignore")

    df["Route"] = df["Origin"].astype(str) + "_" + df["Dest"].astype(str)


    final_columns = [
    'Reporting_Airline',
    'Origin',
    'Dest',
    'Route',
    'ArrDel15',
    'Day',
    'Month',
    'DayOfWeek',
    'IsWeekend',
    'IsFixedHoliday',
    'IsHolidayWindow',
    'DepHour',
    'DepTimeCategory',
    'temp',
    'prcp',
    'rain',
    'snow',
    'wspd',
    'Rain_type',
    'Snow_type',
    'Wind_type',
    ]

    df = df[[col for col in final_columns if col in df.columns]]

    print("Final dataset shape:", df.shape)
    print("Final columns:", df.columns.tolist())

        

    # Save
    df.to_csv(output_path, index=False)
    print("Saved:", output_path)
    print("Final shape:", df.shape)
    print("Target distribution:\n", df["ArrDel15"].value_counts(dropna=False))


    

if __name__ == "__main__":
    main()