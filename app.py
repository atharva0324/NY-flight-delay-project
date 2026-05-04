import pickle
import requests
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

# ── Airport coordinates (for OpenMeteo weather fetch) ──────────────────────
AIRPORT_COORDS = {
    # NY State origins
    "JFK": (40.6413, -73.7781),
    "LGA": (40.7769, -73.8740),
    "BUF": (42.9405, -78.7322),
    "SYR": (43.1112, -76.1063),
    "ROC": (43.1189, -77.6724),
    "ALB": (42.7483, -73.8016),
    "ISP": (40.7952, -73.1002),
    "HPN": (41.0670, -73.7076),
    "SWF": (41.5041, -74.1048),
    "BGM": (42.2082, -75.9798),
    "ELM": (42.1599, -76.8914),
    "IAG": (43.1011, -79.0441),
    "PBG": (44.6509, -73.4681),
    "ITH": (42.4911, -76.4584),
    # Common destinations
    "ORD": (41.9742, -87.9073),
    "MCO": (28.4312, -81.3081),
    "ATL": (33.6367, -84.4281),
    "BOS": (42.3656, -71.0096),
    "DCA": (38.8521, -77.0377),
    "CLT": (35.2140, -80.9431),
    "MIA": (25.7959, -80.2870),
    "FLL": (26.0742, -80.1506),
    "LAX": (33.9425, -118.4081),
    "DFW": (32.8998, -97.0403),
    "DTW": (42.2162, -83.3554),
    "PBI": (26.6832, -80.0956),
    "DEN": (39.8561, -104.6737),
    "TPA": (27.9755, -82.5332),
    "SFO": (37.6213, -122.3790),
    "BNA": (36.1263, -86.6774),
    "BWI": (39.1774, -76.6684),
    "RDU": (35.8776, -78.7875),
    "LAS": (36.0840, -115.1537),
    "CMH": (39.9980, -82.8919),
    "PIT": (40.4915, -80.2329),
    "IAH": (29.9902, -95.3368),
    "SJU": (18.4373, -66.0041),
    "MDW": (41.7868, -87.7522),
    "IND": (39.7173, -86.2944),
    "RSW": (26.5362, -81.7552),
    "MSP": (44.8848, -93.2223),
    "PHX": (33.4373, -112.0078),
    "SEA": (47.4502, -122.3088),
    "CLE": (41.4117, -81.8498),
    "STL": (38.7487, -90.3700),
    "MSY": (29.9934, -90.2580),
    "SAN": (32.7338, -117.1933),
    "SLC": (40.7884, -111.9778),
    "AUS": (30.1975, -97.6664),
    "CHS": (32.8986, -80.0405),
    "GSO": (36.0978, -79.9373),
    "RIC": (37.5052, -77.3197),
    "CVG": (39.0488, -84.6678),
    "MCI": (39.2976, -94.7139),
    "JAX": (30.4941, -81.6879),
    "SAV": (32.1276, -81.2021),
    "PWM": (43.6462, -70.3093),
    "SRQ": (27.3954, -82.5544),
    "DAL": (32.8471, -96.8518),
    "EWR": (40.6925, -74.1687),
}

AIRPORT_DISPLAY = {
    "JFK": "John F. Kennedy (JFK) — New York City",
    "LGA": "LaGuardia (LGA) — New York City",
    "BUF": "Buffalo Niagara (BUF)",
    "SYR": "Syracuse Hancock (SYR)",
    "ROC": "Greater Rochester (ROC)",
    "ALB": "Albany (ALB)",
    "ISP": "Long Island MacArthur (ISP)",
    "HPN": "Westchester County (HPN)",
    "SWF": "Stewart (SWF) — Newburgh",
    "BGM": "Greater Binghamton (BGM)",
    "ELM": "Elmira/Corning (ELM)",
    "IAG": "Niagara Falls (IAG)",
    "PBG": "Plattsburgh (PBG)",
    "ITH": "Ithaca Tompkins (ITH)",
}

AIRLINE_DISPLAY = {
    "9E": "Endeavor Air (9E)",
    "AA": "American Airlines (AA)",
    "AS": "Alaska Airlines (AS)",
    "B6": "JetBlue Airways (B6)",
    "DL": "Delta Air Lines (DL)",
    "F9": "Frontier Airlines (F9)",
    "G4": "Allegiant Air (G4)",
    "HA": "Hawaiian Airlines (HA)",
    "MQ": "Envoy Air / American Eagle (MQ)",
    "NK": "Spirit Airlines (NK)",
    "OH": "PSA Airlines (OH)",
    "OO": "SkyWest Airlines (OO)",
    "UA": "United Airlines (UA)",
    "WN": "Southwest Airlines (WN)",
    "YX": "Republic Airways (YX)",
}

# Historical delay rates computed from training data
AIRLINE_DELAY_RATES = {
    "9E": 0.2006, "AA": 0.2362, "AS": 0.2848, "B6": 0.2680,
    "DL": 0.2309, "F9": 0.3055, "G4": 0.2945, "HA": 0.3132,
    "MQ": 0.2276, "NK": 0.2039, "OH": 0.3131, "OO": 0.2252,
    "UA": 0.2473, "WN": 0.2185, "YX": 0.2248,
}

ORIGIN_DELAY_RATES = {
    "ALB": 0.2172, "BGM": 0.1538, "BUF": 0.2095, "ELM": 0.1844,
    "HPN": 0.2240, "IAG": 0.2953, "ISP": 0.2327, "ITH": 0.1270,
    "JFK": 0.2401, "LGA": 0.2507, "PBG": 0.2727, "ROC": 0.2246,
    "SWF": 0.2803, "SYR": 0.2313,
}

GLOBAL_DELAY_RATE = 0.2402

TOP_DESTINATIONS = [
    "ORD", "MCO", "ATL", "BOS", "DCA", "CLT", "MIA", "FLL", "LAX", "DFW",
    "DTW", "PBI", "DEN", "TPA", "SFO", "BNA", "BWI", "RDU", "LAS", "CMH",
    "PIT", "IAH", "SJU", "MDW", "IND", "RSW", "MSP", "PHX", "SEA", "CLE",
    "STL", "MSY", "SAN", "SLC", "AUS", "CHS", "GSO", "RIC", "CVG", "MCI",
    "JAX", "SAV", "PWM", "SRQ", "DAL", "EWR",
]


# ── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(ROOT / "models" / "xgb_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_route_rates() -> dict:
    data_path = ROOT / "data" / "processed" / "ny_flights_clean.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, low_memory=False)
        df = df.dropna(subset=["ArrDel15"])
        return df.groupby("Route")["ArrDel15"].mean().to_dict()
    return {}


# ── Feature engineering helpers ─────────────────────────────────────────────
def classify_rain(rain: float) -> str:
    if rain > 6:  return "Heavy"
    if rain > 2:  return "Moderate"
    if rain > 0:  return "Normal"
    return "None"


def classify_snow(snow: float) -> str:
    if snow > 5:  return "Heavy"
    if snow > 2:  return "Moderate"
    if snow > 0:  return "Normal"
    return "None"


def classify_wind(wspd: float) -> str:
    if wspd > 35: return "Very_Windy"
    if wspd >= 20: return "Windy"
    return "Normal"


def get_dep_time_category(hour: int) -> str:
    if 5 <= hour <= 11:  return "Morning"
    if 12 <= hour <= 16: return "Afternoon"
    if 17 <= hour <= 21: return "Evening"
    return "Late Night"


# Fixed US federal holidays (month, day)
_FIXED_HOLIDAYS = {(1, 1), (7, 4), (11, 11), (12, 25)}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d, count = date(year, month, 1), 0
    while d.month == month:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += timedelta(days=1)
    return None  # type: ignore[return-value]


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _floating_holidays(year: int) -> set:
    return {
        _nth_weekday(year, 1, 0, 3),   # MLK Day
        _nth_weekday(year, 2, 0, 3),   # Presidents Day
        _last_weekday(year, 5, 0),     # Memorial Day
        _nth_weekday(year, 9, 0, 1),   # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
    }


def is_holiday(d: date) -> bool:
    return (d.month, d.day) in _FIXED_HOLIDAYS or d in _floating_holidays(d.year)


def is_holiday_window(d: date, window: int = 3) -> bool:
    return any(is_holiday(d + timedelta(days=i)) for i in range(-window, window + 1))


# ── Weather fetch ────────────────────────────────────────────────────────────
def fetch_weather(lat: float, lon: float, target_date: date, dep_hour: int):
    if target_date < date.today():
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": target_date.isoformat(),
        "end_date": target_date.isoformat(),
        "hourly": "temperature_2m,precipitation,rain,snowfall,wind_speed_10m",
        "timezone": "America/New_York",
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if "hourly" not in data:
            return None
        h = data["hourly"]
        return {
            "temp": h["temperature_2m"][dep_hour],
            "prcp": h["precipitation"][dep_hour],
            "rain": h["rain"][dep_hour],
            "snow": h["snowfall"][dep_hour],
            "wspd": h["wind_speed_10m"][dep_hour],
        }
    except Exception:
        return None


# ── App layout ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NY Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header
st.title("✈️ NY Flight Delay Predictor")
st.caption(
    "Predicts whether your flight departing from a **New York State airport** "
    "will arrive **15+ minutes late**, using an XGBoost model trained on BTS "
    "on-time performance data + real weather."
)
st.divider()

# ── Input columns ────────────────────────────────────────────────────────────
col_flight, col_weather = st.columns(2, gap="large")

with col_flight:
    st.subheader("Flight Details")

    airline_code = st.selectbox(
        "Airline",
        options=list(AIRLINE_DISPLAY.keys()),
        format_func=lambda x: AIRLINE_DISPLAY[x],
        index=list(AIRLINE_DISPLAY.keys()).index("DL"),
    )

    origin_code = st.selectbox(
        "Origin Airport (NY State)",
        options=list(AIRPORT_DISPLAY.keys()),
        format_func=lambda x: AIRPORT_DISPLAY[x],
        index=list(AIRPORT_DISPLAY.keys()).index("JFK"),
    )

    dest_col1, dest_col2 = st.columns([2, 1])
    with dest_col1:
        dest_select = st.selectbox(
            "Destination Airport",
            options=[""] + TOP_DESTINATIONS,
            help="Pick a common destination or type a code in the field to the right",
        )
    with dest_col2:
        dest_custom = st.text_input(
            "Custom code",
            max_chars=3,
            placeholder="e.g. SFO",
        ).upper().strip()

    dest_code = dest_custom if dest_custom else dest_select

    flight_date = st.date_input(
        "Flight Date",
        value=date.today() + timedelta(days=1),
        min_value=date(2020, 1, 1),
        max_value=date.today() + timedelta(days=16),
    )

    dep_time = st.time_input(
        "Scheduled Departure Time",
        value=datetime.strptime("08:00", "%H:%M").time(),
        step=3600,
    )

with col_weather:
    st.subheader("Weather at Destination")

    if st.button("Auto-Fetch Weather from OpenMeteo", use_container_width=True):
        if not dest_code or len(dest_code) != 3:
            st.warning("Select a destination airport first, then fetch weather.")
        elif dest_code not in AIRPORT_COORDS:
            st.warning(f"No coordinates on file for {dest_code}. Enter weather manually below.")
        else:
            lat, lon = AIRPORT_COORDS[dest_code]
            with st.spinner(f"Fetching weather for {dest_code}…"):
                wx = fetch_weather(lat, lon, flight_date, dep_time.hour)
            if wx:
                st.session_state["wx"] = wx
                st.success(f"Weather loaded for {dest_code} — values updated below.")
            else:
                st.warning("Could not fetch weather. Enter values manually below.")

    defaults = st.session_state.get("wx", {
        "temp": 15.0, "prcp": 0.0, "rain": 0.0, "snow": 0.0, "wspd": 12.0,
    })

    w1, w2 = st.columns(2)
    with w1:
        temp = st.number_input("Temperature (°C)",         value=float(defaults["temp"]), step=0.5)
        rain = st.number_input("Rain (mm)",                 value=float(defaults["rain"]), min_value=0.0, step=0.1)
        snow = st.number_input("Snowfall (cm)",             value=float(defaults["snow"]), min_value=0.0, step=0.1)
    with w2:
        prcp = st.number_input("Total Precipitation (mm)", value=float(defaults["prcp"]), min_value=0.0, step=0.1)
        wspd = st.number_input("Wind Speed (km/h)",        value=float(defaults["wspd"]), min_value=0.0, step=0.5)

    with st.expander("Weather classification (auto-computed)"):
        st.markdown(f"""
| Feature | Value |
|---|---|
| Rain type | `{classify_rain(rain)}` |
| Snow type | `{classify_snow(snow)}` |
| Wind type | `{classify_wind(wspd)}` |
| Dep. time category | `{get_dep_time_category(dep_time.hour)}` |
        """)

st.divider()

# ── Predict ──────────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Delay", type="primary", use_container_width=True)

if predict_clicked:
    if not dest_code or len(dest_code) != 3:
        st.error("Please select or enter a valid 3-letter destination airport code.")
        st.stop()

    model = load_model()
    route_rates = load_route_rates()

    d = flight_date
    route = f"{origin_code}_{dest_code}"
    dep_hour = dep_time.hour
    dow = d.isoweekday()  # 1=Mon … 7=Sun

    airline_rate = AIRLINE_DELAY_RATES.get(airline_code, GLOBAL_DELAY_RATE)
    origin_rate  = ORIGIN_DELAY_RATES.get(origin_code, GLOBAL_DELAY_RATE)
    route_rate   = route_rates.get(route, round((airline_rate + origin_rate) / 2, 4))

    row = {
        "Reporting_Airline":   airline_code,
        "Origin":              origin_code,
        "Dest":                dest_code,
        "Route":               route,
        "Day":                 d.day,
        "Month":               d.month,
        "DayOfWeek":           dow,
        "IsWeekend":           int(dow >= 6),
        "IsFixedHoliday":      int(is_holiday(d)),
        "IsHolidayWindow":     int(is_holiday_window(d)),
        "DepHour":             dep_hour,
        "DepTimeCategory":     get_dep_time_category(dep_hour),
        "temp":                temp,
        "prcp":                prcp,
        "rain":                rain,
        "snow":                snow,
        "wspd":                wspd,
        "Rain_type":           classify_rain(rain),
        "Snow_type":           classify_snow(snow),
        "Wind_type":           classify_wind(wspd),
        "airline_delay_rate":  airline_rate,
        "origin_delay_rate":   origin_rate,
        "Route_Arr_Delay_Rate": route_rate,
    }

    df_input = pd.DataFrame([row])
    pred = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0][1]) if hasattr(model, "predict_proba") else 0.5

    # ── Result card ───────────────────────────────────────────────────────
    st.subheader("Prediction Result")
    res_col, prob_col, detail_col = st.columns([1, 1, 2], gap="medium")

    with res_col:
        if pred == 1:
            st.error("### LIKELY DELAYED")
            st.markdown("The model predicts this flight will arrive **≥ 15 min late**.")
        else:
            st.success("### LIKELY ON TIME")
            st.markdown("The model predicts this flight will arrive **on schedule**.")

    with prob_col:
        st.metric("Delay Probability", f"{prob * 100:.1f}%")
        bar_color = "red" if prob >= 0.5 else "green"
        st.progress(prob, text=f"{prob * 100:.1f}% chance of delay")

        # Risk level label
        if prob < 0.20:
            risk = "🟢 Low risk"
        elif prob < 0.40:
            risk = "🟡 Moderate risk"
        elif prob < 0.60:
            risk = "🟠 Elevated risk"
        else:
            risk = "🔴 High risk"
        st.markdown(f"**{risk}**")

    with detail_col:
        st.markdown("**Flight summary**")
        st.markdown(f"""
- **Route:** {airline_code} {origin_code} → {dest_code}
- **Date:** {d.strftime("%A, %B %d, %Y")}
- **Departure:** {dep_time.strftime("%H:%M")} ({get_dep_time_category(dep_hour)})
- **Weekend:** {"Yes" if dow >= 6 else "No"} &nbsp;|&nbsp; **Holiday:** {"Yes" if is_holiday(d) else "No"} &nbsp;|&nbsp; **Holiday window:** {"Yes" if is_holiday_window(d) else "No"}
- **Weather:** {temp}°C, rain {rain} mm, snow {snow} cm, wind {wspd} km/h
- **Conditions:** {classify_rain(rain)} rain · {classify_snow(snow)} snow · {classify_wind(wspd)}
- **Hist. airline delay rate:** {airline_rate:.1%} &nbsp;|&nbsp; **Origin delay rate:** {origin_rate:.1%}
        """)

    with st.expander("Full feature vector sent to model"):
        st.dataframe(
            df_input.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

    st.caption(
        "Model: XGBoost trained on BTS on-time performance data (NY State airports, 2024–2025). "
        "Accuracy ~76%, ROC-AUC ~0.82. Predictions are probabilistic — not a guarantee."
    )
