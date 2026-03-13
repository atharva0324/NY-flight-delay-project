import os
import pandas as pd
import matplotlib.pyplot as plt

def save_bar(series, title, xlabel, ylabel, outpath, top_n=None):
    s = series.dropna()
    if top_n is not None:
        s = s.sort_values(ascending=False).head(top_n)
    plt.figure()
    s.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def save_line(series, title, xlabel, ylabel, outpath):
    s = series.dropna().sort_index()
    plt.figure()
    s.plot(kind="line", marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    # 0) Paths
    input_csv = "data/processed/ny_flights_clean.csv"   
    out_dir = "outputs/eda"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(input_csv)
    print("Loaded:", input_csv)
    print("Shape:", df.shape)

    # 2) Columns + datatypes
    print("\n--- Columns ---")
    print(list(df.columns))
    print("\n--- Dtypes ---")
    print(df.dtypes)

    # 3) Missing values table (count + %)
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2)
    }).sort_values("missing_count", ascending=False)
    missing.to_csv(os.path.join(out_dir, "missing_summary.csv"))
    print("\nSaved missing summary -> outputs/eda/missing_summary.csv")

    # 4) Target distribution
    if "ArrDel15" not in df.columns:
        raise ValueError("ArrDel15 column not found in cleaned file. Check your data_cleaning output.")

    target_counts = df["ArrDel15"].value_counts(dropna=False)
    target_pct = (df["ArrDel15"].value_counts(normalize=True, dropna=False) * 100).round(2)

    target_table = pd.DataFrame({"count": target_counts, "pct": target_pct})
    target_table.to_csv(os.path.join(out_dir, "target_distribution.csv"))
    print("\n--- Target Distribution (ArrDel15) ---")
    print(target_table)

    
    save_bar(target_counts, "Target Distribution (ArrDel15)", "ArrDel15", "Count",
             os.path.join(out_dir, "target_distribution.png"))

    # Helper: delay rate = mean of ArrDel15 (works since it’s 0/1)
    # 5) Delay rate by Month
    if "Month" in df.columns:
        month_rate = df.groupby("Month")["ArrDel15"].mean()
        month_rate.to_csv(os.path.join(out_dir, "delay_rate_by_month.csv"))
        save_line(month_rate, "Delay Rate by Month", "Month", "Delay Rate",
                  os.path.join(out_dir, "delay_rate_by_month.png"))
        print("\nSaved month EDA.")

    # 6) Delay rate by DayOfWeek or DayName
    if "DayOfWeek" in df.columns:
        dow_rate = df.groupby("DayOfWeek")["ArrDel15"].mean()
        dow_rate.to_csv(os.path.join(out_dir, "delay_rate_by_dayofweek.csv"))
        save_bar(dow_rate, "Delay Rate by Day of Week", "DayOfWeek", "Delay Rate",
                 os.path.join(out_dir, "delay_rate_by_dayofweek.png"))
        print("Saved DayOfWeek EDA.")
    elif "DayName" in df.columns:
        day_rate = df.groupby("DayName")["ArrDel15"].mean()
        day_rate = day_rate.reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        day_rate.to_csv(os.path.join(out_dir, "delay_rate_by_dayname.csv"))
        save_bar(day_rate, "Delay Rate by Day Name", "DayName", "Delay Rate",
                 os.path.join(out_dir, "delay_rate_by_dayname.png"))
        print("Saved DayName EDA.")

    # 7) Delay rate by DepHour
    if "DepHour" in df.columns:
        hour_rate = df.groupby("DepHour")["ArrDel15"].mean()
        hour_rate.to_csv(os.path.join(out_dir, "delay_rate_by_dephour.csv"))
        save_line(hour_rate, "Delay Rate by Departure Hour", "DepHour", "Delay Rate",
                  os.path.join(out_dir, "delay_rate_by_dephour.png"))
        print("Saved DepHour EDA.")

    # 8) Delay rate by airline (top 10)
    if "Reporting_Airline" in df.columns:
        airline_rate = df.groupby("Reporting_Airline")["ArrDel15"].mean().sort_values(ascending=False)
        airline_rate.to_csv(os.path.join(out_dir, "delay_rate_by_airline.csv"))
        save_bar(airline_rate, "Top 10 Airlines by Delay Rate", "Airline", "Delay Rate",
                 os.path.join(out_dir, "delay_rate_by_airline_top10.png"), top_n=10)
        print("Saved airline EDA.")

    # 9) Delay rate by Origin (top 10)
    if "Origin" in df.columns:
        origin_rate = df.groupby("Origin")["ArrDel15"].mean().sort_values(ascending=False)
        origin_rate.to_csv(os.path.join(out_dir, "delay_rate_by_origin.csv"))
        save_bar(origin_rate, "Top 10 Origin Airports by Delay Rate", "Origin", "Delay Rate",
                 os.path.join(out_dir, "delay_rate_by_origin_top10.png"), top_n=10)
        print("Saved origin EDA.")

    # 10) Delay rate by DestState (top 10)
    if "DestState" in df.columns:
        dest_state_rate = df.groupby("DestState")["ArrDel15"].mean().sort_values(ascending=False)
        dest_state_rate.to_csv(os.path.join(out_dir, "delay_rate_by_deststate.csv"))
        save_bar(dest_state_rate, "Top 10 Destination States by Delay Rate", "DestState", "Delay Rate",
                 os.path.join(out_dir, "delay_rate_by_deststate_top10.png"), top_n=10)
        print("Saved destination state EDA.")

   
    print("\n EDA complete. Check outputs in:", out_dir)

if __name__ == "__main__":
    main()