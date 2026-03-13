import pandas as pd
from sklearn.model_selection import train_test_split

def add_target_mean_features(train_df, test_df, target_col="ArrDel15"):
    global_mean = train_df[target_col].mean()

    airline_map = train_df.groupby("Reporting_Airline")[target_col].mean()
    train_df["airline_delay_rate"] = train_df["Reporting_Airline"].map(airline_map)
    test_df["airline_delay_rate"] = test_df["Reporting_Airline"].map(airline_map).fillna(global_mean)

    origin_map = train_df.groupby("Origin")[target_col].mean()
    train_df["origin_delay_rate"] = train_df["Origin"].map(origin_map)
    test_df["origin_delay_rate"] = test_df["Origin"].map(origin_map).fillna(global_mean)

    route_map = train_df.groupby("Route")[target_col].mean()
    train_df["Route_Arr_Delay_Rate"] = train_df["Route"].map(route_map)
    test_df["Route_Arr_Delay_Rate"] = test_df["Route"].map(route_map).fillna(global_mean)

    return train_df, test_df

def load_split_data(path="data/processed/ny_flights_clean.csv"):
    df = pd.read_csv(path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        stratify=df["ArrDel15"],
        random_state=42
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df, test_df = add_target_mean_features(train_df, test_df)

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_split_data()
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print(train_df.columns.tolist())