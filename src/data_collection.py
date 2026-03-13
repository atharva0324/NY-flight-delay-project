import pandas as pd
import os

def main():
    print("Starting data collection...")

    df_2024_12 = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_12.csv")

    df_2025_1  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_1.csv")
    df_2025_2  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_2.csv")
    df_2025_3  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_3.csv")
    df_2025_4  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_4.csv")
    df_2025_5  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_5.csv")
    df_2025_6  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_6.csv")
    df_2025_7  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_7.csv")
    df_2025_8  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_8.csv")
    df_2025_9  = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_9.csv")
    df_2025_10 = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_10.csv")
    df_2025_11 = pd.read_csv("data/raw/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2025_11.csv")

    combined_df = pd.concat([
        df_2025_1, df_2025_2, df_2025_3, df_2025_4, df_2025_5,
        df_2025_6, df_2025_7, df_2025_8, df_2025_9, df_2025_10,df_2025_11,df_2024_12
    ], ignore_index=True)

    print("Total rows after combining:", combined_df.shape)

    output_path = "data/processed/combined_flights.csv"

    print("Saving combined CSV in chunks...")

    chunk_size = 200_000
    total_rows = len(combined_df)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        mode = "w" if start == 0 else "a"
        header = True if start == 0 else False

        combined_df.iloc[start:end].to_csv(output_path, mode=mode, header=header, index=False)

        print(f"Saved rows {start:,} to {end:,} / {total_rows:,}")

    print("Done saving:", output_path)

    print("Data collection completed.")
if __name__ == "__main__":
    main()