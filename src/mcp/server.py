from pathlib import Path
import pickle
import pandas as pd
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flight-delay-model")

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "xgb_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

EXPECTED_FIELDS = [
    "Reporting_Airline", "Origin", "Dest", "Route", "Day", "Month",
    "DayOfWeek", "IsWeekend", "IsFixedHoliday", "IsHolidayWindow",
    "DepHour", "DepTimeCategory", "temp", "prcp", "rain", "snow",
    "wspd", "Rain_type", "Snow_type", "Wind_type",
    "airline_delay_rate", "origin_delay_rate", "Route_Arr_Delay_Rate"
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
        if Month < 1 or Month > 12:
            return {"error": "Month must be between 1 and 12"}
        if Day < 1 or Day > 31:
            return {"error": "Day must be between 1 and 31"}
        if DayOfWeek < 1 or DayOfWeek > 7:
            return {"error": "DayOfWeek must be between 1 and 7"}
        if IsWeekend not in [0, 1]:
            return {"error": "IsWeekend must be 0 or 1"}
        if IsFixedHoliday not in [0, 1]:
            return {"error": "IsFixedHoliday must be 0 or 1"}
        if IsHolidayWindow not in [0, 1]:
            return {"error": "IsHolidayWindow must be 0 or 1"}
        if DepHour < 0 or DepHour > 23:
            return {"error": "DepHour must be between 0 and 23"}

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

        if set(row.keys()) != set(EXPECTED_FIELDS):
            return {"error": "Mismatch in expected input fields"}

        df = pd.DataFrame([row])
        pred = model.predict(df)[0]

        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(df)[0][1]

        return {
            "prediction": int(pred),
            "delay_probability": float(round(prob, 4)) if prob is not None else None
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()