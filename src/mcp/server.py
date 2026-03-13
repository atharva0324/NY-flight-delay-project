import pickle
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flight-delay-model")


with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)
print("Model loaded successfully")



EXPECTED_FIELDS = [
    "Reporting_Airline",
    "Origin",
    "Dest",
    "Route",
    "Day",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "IsFixedHoliday",
    "IsHolidayWindow",
    "DepHour",
    "DepTimeCategory",
    "temp",
    "prcp",
    "rain",
    "snow",
    "wspd",
    "Rain_type",
    "Snow_type",
    "Wind_type",
    "airline_delay_rate",
    "origin_delay_rate",
    "Route_Arr_Delay_Rate",
]


@mcp.tool()
def predict(
    Reporting_Airline: str,
    Origin: str,
    Dest: str,
    Route: str,
    Day: int,
    Month: int,
    DayOfWeek: int,
    IsWeekend: int,
    IsFixedHoliday: int,
    IsHolidayWindow: int,
    DepHour: int,
    DepTimeCategory: str,
    temp: float,
    prcp: float,
    rain: float,
    snow: float,
    wspd: float,
    Rain_type: str,
    Snow_type: str,
    Wind_type: str,
    airline_delay_rate: float,
    origin_delay_rate: float,
    Route_Arr_Delay_Rate: float,
) -> dict:
    try:
        row = {
            "Reporting_Airline": Reporting_Airline,
            "Origin": Origin,
            "Dest": Dest,
            "Route": Route,
            "Day": Day,
            "Month": Month,
            "DayOfWeek": DayOfWeek,
            "IsWeekend": IsWeekend,
            "IsFixedHoliday": IsFixedHoliday,
            "IsHolidayWindow": IsHolidayWindow,
            "DepHour": DepHour,
            "DepTimeCategory": DepTimeCategory,
            "temp": temp,
            "prcp": prcp,
            "rain": rain,
            "snow": snow,
            "wspd": wspd,
            "Rain_type": Rain_type,
            "Snow_type": Snow_type,
            "Wind_type": Wind_type,
            "airline_delay_rate": airline_delay_rate,
            "origin_delay_rate": origin_delay_rate,
            "Route_Arr_Delay_Rate": Route_Arr_Delay_Rate,
        }

        df = pd.DataFrame([row])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(pred),
            "delay_probability": float(round(prob, 4))
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print("Starting MCP server...")
    mcp.run()