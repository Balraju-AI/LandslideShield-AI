import os
import json
from datetime import datetime, date, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LandslideShield AI",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = "model/landslide_model.pkl"
METRICS_PATH = "model/metrics.json"
LOCATION_PATH = "data/india_locations.csv"


# ============================================================
# SESSION STATE
# ============================================================

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "sms_sent_keys" not in st.session_state:
    st.session_state.sms_sent_keys = set()

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# SIMPLE STYLING
# No HTML hero section
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


model = load_model()


# ============================================================
# LOAD METRICS
# ============================================================

@st.cache_data
def load_metrics():

    if not os.path.exists(METRICS_PATH):
        return {}

    try:
        with open(
            METRICS_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return {}


metrics = load_metrics()


# ============================================================
# LOAD LOCATION DATABASE
# ============================================================

@st.cache_data
def load_locations():

    if not os.path.exists(LOCATION_PATH):

        return pd.DataFrame(
            columns=[
                "state",
                "district",
                "city",
                "latitude",
                "longitude"
            ]
        )

    try:

        df = pd.read_csv(
            LOCATION_PATH
        )

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

        required = [
            "state",
            "district",
            "city",
            "latitude",
            "longitude"
        ]

        for column in required:

            if column not in df.columns:
                df[column] = ""

        return df

    except Exception:

        return pd.DataFrame(
            columns=[
                "state",
                "district",
                "city",
                "latitude",
                "longitude"
            ]
        )


locations = load_locations()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Summer"

    if month in [6, 7, 8, 9]:
        return "Monsoon"

    return "Post-Monsoon"


def clamp(value, low=0, high=100):

    return max(
        low,
        min(
            high,
            float(value)
        )
    )


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


def risk_level(score):

    if score >= 85:
        return "CRITICAL"

    if score >= 70:
        return "HIGH"

    if score >= 45:
        return "MODERATE"

    return "LOW"


def signal_status(score):

    if score >= 70:
        return "HIGH"

    if score >= 35:
        return "MODERATE"

    return "LOW"


def risk_message(level):

    messages = {

        "CRITICAL":
            "Multiple strong warning signals are present. Immediate attention and continuous monitoring are recommended.",

        "HIGH":
            "Elevated landslide concern detected. Close monitoring and precautionary action are recommended.",

        "MODERATE":
            "Some warning indicators are present. Continue monitoring local conditions.",

        "LOW":
            "No strong combined warning signal was detected from the available inputs."
    }

    return messages.get(level, "")


# ============================================================
# WEATHER API
# ============================================================

@st.cache_data(ttl=600)
def get_weather(latitude, longitude):

    url = "https://api.open-meteo.com/v1/forecast"

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current":
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "snowfall,"
            "wind_speed_10m,"
            "weather_code",

        "hourly":
            "precipitation,"
            "rain,"
            "snowfall,"
            "wind_speed_10m",

        "past_hours": 72,

        "forecast_hours": 24,

        "timezone": "auto"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        return response.json()

    except Exception:

        return None


# ============================================================
# PROCESS WEATHER
# ============================================================

def process_weather(data):

    if not data:
        return None

    current = data.get(
        "current",
        {}
    )

    result = {

        "temperature":
            safe_float(
                current.get(
                    "temperature_2m",
                    0
                )
            ),

        "humidity":
            safe_float(
                current.get(
                    "relative_humidity_2m",
                    0
                )
            ),

        "precipitation":
            safe_float(
                current.get(
                    "precipitation",
                    0
                )
            ),

        "rain":
            safe_float(
                current.get(
                    "rain",
                    0
                )
            ),

        "snowfall":
            safe_float(
                current.get(
                    "snowfall",
                    0
                )
            ),

        "wind":
            safe_float(
                current.get(
                    "wind_speed_10m",
                    0
                )
            )
    }

    hourly = data.get(
        "hourly",
        {}
    )

    rain_values = hourly.get(
        "rain",
        []
    )

    precipitation_values = hourly.get(
        "precipitation",
        []
    )

    snowfall_values = hourly.get(
        "snowfall",
        []
    )

    wind_values = hourly.get(
        "wind_speed_10m",
        []
    )

    result["rain_72h"] = float(
        np.nansum(
            rain_values
        )
    )

    result["precipitation_72h"] = float(
        np.nansum(
            precipitation_values
        )
    )

    result["snowfall_72h"] = float(
        np.nansum(
            snowfall_values
        )
    )

    if wind_values:

        result["max_wind"] = float(
            np.nanmax(
                wind_values
            )
        )

    else:

        result["max_wind"] = result["wind"]

    return result


# ============================================================
# EARTHQUAKE API
# ============================================================

@st.cache_data(ttl=600)
def get_earthquakes(latitude, longitude):

    end_time = datetime.utcnow()

    start_time = (
        end_time -
        timedelta(days=7)
    )

    url = (
        "https://earthquake.usgs.gov/"
        "fdsnws/event/1/query"
    )

    params = {

        "format": "geojson",

        "starttime":
            start_time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

        "endtime":
            end_time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),

        "latitude": latitude,

        "longitude": longitude,

        "maxradiuskm": 300,

        "minmagnitude": 2.5
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        features = data.get(
            "features",
            []
        )

        magnitudes = []

        for item in features:

            properties = item.get(
                "properties",
                {}
            )

            magnitude = properties.get(
                "mag"
            )

            if magnitude is not None:
                magnitudes.append(
                    float(magnitude)
                )

        return {

            "count":
                len(features),

            "max_magnitude":
                max(magnitudes)
                if magnitudes
                else 0
        }

    except Exception:

        return {

            "count":
                0,

            "max_magnitude":
                0
        }


# ============================================================
# RISK INDICES
# ============================================================

def rainfall_index(weather):

    if not weather:
        return 0

    current_rain = max(
        weather.get(
            "rain",
            0
        ),
        weather.get(
            "precipitation",
            0
        )
    )

    rain_72h = weather.get(
        "rain_72h",
        0
    )

    score = (

        min(
            current_rain / 50,
            1
        ) * 40

        +

        min(
            rain_72h / 200,
            1
        ) * 60
    )

    return clamp(score)


def storm_index(weather):

    if not weather:
        return 0

    max_wind = weather.get(
        "max_wind",
        0
    )

    return clamp(
        (max_wind / 100) * 100
    )


def snow_index(weather):

    if not weather:
        return 0

    snowfall = weather.get(
        "snowfall_72h",
        0
    )

    return clamp(
        (snowfall / 100) * 100
    )


def seismic_index(earthquake):

    if not earthquake:
        return 0

    count = earthquake.get(
        "count",
        0
    )

    magnitude = earthquake.get(
        "max_magnitude",
        0
    )

    score = (

        min(
            count / 5,
            1
        ) * 40

        +

        min(
            magnitude / 6,
            1
        ) * 60
    )

    return clamp(score)


# ============================================================
# ML PREDICTION
# ============================================================

def make_prediction(
    latitude,
    longitude,
    state,
    selected_date
):

    if model is None:

        return (
            None,
            None,
            None
        )

    year = selected_date.year
    month = selected_date.month
    day = selected_date.day

    day_of_year = (
        selected_date
        .timetuple()
        .tm_yday
    )

    season = get_season(
        month
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    data = pd.DataFrame(
        [{
            "latitude":
                latitude,

            "longitude":
                longitude,

            "year":
                year,

            "month":
                month,

            "day":
                day,

            "day_of_year":
                day_of_year,

            "season":
                season,

            "state":
                state,

            "month_sin":
                month_sin,

            "month_cos":
                month_cos
        }]
    )

    try:

        prediction = model.predict(
            data
        )[0]

        probabilities = None

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = model.predict_proba(
                data
            )[0]

        return (
            str(prediction),
            probabilities,
            data
        )

    except Exception as error:

        st.error(
            "ML prediction failed."
        )

        st.code(
            str(error)
        )

        return (
            None,
            None,
            data
        )


# ============================================================
# MODEL CLASSES
# ============================================================

def get_model_classes():

    if model is None:
        return []

    try:
        return model.classes_
    except Exception:
        pass

    try:
        return (
            model.named_steps[
                "classifier"
            ].classes_
        )
    except Exception:
        pass

    try:
        return (
            model.named_steps[
                "model"
            ].classes_
        )
    except Exception:
        return []


# ============================================================
# TWILIO SMS
# ============================================================

def send_sms_alert(
    level,
    score,
    state,
    district,
    city,
    prediction
):

    if level not in [
        "HIGH",
        "CRITICAL"
    ]:

        return (
            False,
            "SMS is only used for HIGH or CRITICAL risk."
        )

    try:

        account_sid = st.secrets[
            "TWILIO_ACCOUNT_SID"
        ]

        auth_token = st.secrets[
            "TWILIO_AUTH_TOKEN"
        ]

        twilio_phone = st.secrets[
            "TWILIO_PHONE"
        ]

        receiver_phone = st.secrets[
            "RECEIVER_PHONE"
        ]

    except Exception:

        return (
            False,
            "Twilio credentials are missing from .streamlit/secrets.toml"
        )

    try:

        from twilio.rest import Client

        client = Client(
            account_sid,
            auth_token
        )

        if level == "CRITICAL":

            title = (
                "CRITICAL LANDSLIDE ALERT"
            )

        else:

            title = (
                "HIGH LANDSLIDE WARNING"
            )

        message_body = (
            "LANDSLIDESHIELD AI\n\n"
            f"{title}\n\n"
            f"Location: {city}, {district}, {state}\n"
            f"Concern Index: {score:.1f}/100\n"
            f"Predicted Trigger: {prediction}\n\n"
            "Please monitor local conditions "
            "and follow official safety guidance."
        )

        message = client.messages.create(
            body=message_body,
            from_=twilio_phone,
            to=receiver_phone
        )

        return (
            True,
            message.sid
        )

    except ImportError:

        return (
            False,
            "Twilio is not installed. Run: pip install twilio"
        )

    except Exception as error:

        return (
            False,
            str(error)
        )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌋 LandslideShield AI"
)

st.subheader(
    "Multi-Factor Landslide Risk Intelligence & Early Warning"
)

st.caption(
    "AI + Live Weather + Seismic Context + Automatic SMS Alerts"
)


if model is not None:

    st.success(
        "🟢 SYSTEM ONLINE — AI MODEL LOADED"
    )

else:

    st.error(
        "🔴 MODEL NOT FOUND — RUN train_model.py"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ System Status"
    )

    if model is not None:

        st.success(
            "AI Model: Loaded"
        )

    else:

        st.error(
            "AI Model: Missing"
        )


    if not locations.empty:

        st.info(
            f"Monitoring Areas: "
            f"{len(locations)}"
        )

        st.info(
            f"States: "
            f"{locations['state'].nunique()}"
        )

        st.info(
            f"Districts: "
            f"{locations['district'].nunique()}"
        )


    st.divider()

    st.subheader(
        "📱 Automatic SMS"
    )


    try:

        sms_ready = all(
            key in st.secrets
            for key in [
                "TWILIO_ACCOUNT_SID",
                "TWILIO_AUTH_TOKEN",
                "TWILIO_PHONE",
                "RECEIVER_PHONE"
            ]
        )

    except Exception:

        sms_ready = False


    if sms_ready:

        st.success(
            "SMS Service: READY"
        )

        st.caption(
            "Automatic SMS: HIGH / CRITICAL"
        )

    else:

        st.warning(
            "SMS Service: NOT CONFIGURED"
        )


    st.divider()

    st.subheader(
        "📊 Model Metrics"
    )


    if metrics:

        accuracy = metrics.get(
            "accuracy"
        )

        macro_f1 = metrics.get(
            "macro_f1"
        )


        if accuracy is not None:

            st.metric(
                "Accuracy",
                f"{float(accuracy) * 100:.1f}%"
            )


        if macro_f1 is not None:

            st.metric(
                "Macro F1",
                f"{float(macro_f1) * 100:.1f}%"
            )


    st.divider()

    st.caption(
        "Academic / Research Prototype"
    )


# ============================================================
# OVERVIEW
# ============================================================

st.header(
    "System Overview"
)

m1, m2, m3, m4 = st.columns(4)


m1.metric(
    "ML Engine",
    "Random Forest"
    if model is not None
    else "Unavailable"
)


m2.metric(
    "Live Weather",
    "ACTIVE"
)


m3.metric(
    "Seismic Context",
    "ACTIVE"
)


m4.metric(
    "SMS Alert",
    "READY"
    if sms_ready
    else "OFF"
)


# ============================================================
# LOCATION
# ============================================================

st.divider()

st.header(
    "📍 Monitoring Location"
)


if locations.empty:

    st.error(
        "india_locations.csv could not be loaded."
    )

    st.stop()


c1, c2, c3 = st.columns(3)


with c1:

    states = sorted(
        locations[
            "state"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    state = st.selectbox(
        "State",
        states
    )


with c2:

    state_df = locations[
        locations[
            "state"
        ].astype(str)
        == state
    ]

    districts = sorted(
        state_df[
            "district"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    district = st.selectbox(
        "District",
        districts
    )


with c3:

    district_df = state_df[
        state_df[
            "district"
        ].astype(str)
        == district
    ]

    cities = sorted(
        district_df[
            "city"
        ]
        .dropna()
        .astype(str)
        .unique()
    )

    city = st.selectbox(
        "City / Monitoring Area",
        cities
    )


selected_location = district_df[
    district_df[
        "city"
    ].astype(str)
    == city
].iloc[0]


latitude = safe_float(
    selected_location["latitude"]
)

longitude = safe_float(
    selected_location["longitude"]
)


st.info(
    f"📍 {city}, {district}, {state} | "
    f"Latitude: {latitude:.4f} | "
    f"Longitude: {longitude:.4f}"
)


# ============================================================
# DATE
# ============================================================

st.header(
    "📅 Assessment Date"
)

analysis_date = st.date_input(
    "Select assessment date",
    value=date.today()
)


# ============================================================
# LIVE ENVIRONMENT
# ============================================================

st.divider()

st.header(
    "🌍 Current Environmental Intelligence"
)


use_live_data = st.checkbox(
    "Use live weather and seismic data",
    value=True
)


weather = None
earthquake = None


if use_live_data:

    with st.spinner(
        "Fetching current environmental data..."
    ):

        weather_raw = get_weather(
            latitude,
            longitude
        )

        weather = process_weather(
            weather_raw
        )

        earthquake = get_earthquakes(
            latitude,
            longitude
        )


# ============================================================
# WEATHER DISPLAY
# ============================================================

if weather:

    w1, w2, w3, w4, w5 = st.columns(5)


    w1.metric(
        "Temperature",
        f"{weather['temperature']:.1f} °C"
    )


    w2.metric(
        "Humidity",
        f"{weather['humidity']:.0f}%"
    )


    w3.metric(
        "Current Rain",
        f"{weather['rain']:.1f} mm"
    )


    w4.metric(
        "Rain - 72h",
        f"{weather['rain_72h']:.1f} mm"
    )


    w5.metric(
        "Wind",
        f"{weather['wind']:.1f} km/h"
    )


else:

    st.warning(
        "Live weather data is unavailable."
    )


# ============================================================
# SEISMIC DISPLAY
# ============================================================

if earthquake:

    st.subheader(
        "🌐 Recent Seismic Context"
    )

    e1, e2 = st.columns(2)


    e1.metric(
        "Earthquakes within 300 km / 7 days",
        earthquake["count"]
    )


    e2.metric(
        "Maximum Magnitude",
        f"{earthquake['max_magnitude']:.1f}"
    )


# ============================================================
# RUN ASSESSMENT
# ============================================================

st.divider()

st.header(
    "🚨 Early Warning Assessment"
)


run_analysis = st.button(
    "🚨 RUN COMPLETE RISK ASSESSMENT",
    use_container_width=True
)


# ============================================================
# MAIN ASSESSMENT
# ============================================================

if run_analysis:

    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    prediction, probabilities, model_input = (
        make_prediction(
            latitude,
            longitude,
            state,
            analysis_date
        )
    )


    # --------------------------------------------------------
    # RISK FACTOR SCORES
    # --------------------------------------------------------

    rain_score = rainfall_index(
        weather
    )

    storm_score = storm_index(
        weather
    )

    snow_score = snow_index(
        weather
    )

    seismic_score = seismic_index(
        earthquake
    )


    # --------------------------------------------------------
    # ML CONFIDENCE
    # --------------------------------------------------------

    if probabilities is not None:

        ml_confidence = (
            float(
                np.max(
                    probabilities
                )
            ) * 100
        )

    else:

        ml_confidence = 0.0


    # --------------------------------------------------------
    # COMPOSITE CONCERN INDEX
    # --------------------------------------------------------

    composite = (

        ml_confidence * 0.45

        +

        rain_score * 0.25

        +

        seismic_score * 0.15

        +

        storm_score * 0.10

        +

        snow_score * 0.05
    )


    composite = clamp(
        composite
    )


    level = risk_level(
        composite
    )


    # --------------------------------------------------------
    # UNIQUE ASSESSMENT KEY
    # --------------------------------------------------------

    analysis_key = (
        f"{state}|"
        f"{district}|"
        f"{city}|"
        f"{analysis_date}|"
        f"{level}|"
        f"{round(composite, 1)}|"
        f"{prediction}"
    )


    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    st.session_state.last_result = {

        "risk_level":
            level,

        "concern_index":
            composite,

        "ml_confidence":
            ml_confidence,

        "prediction":
            prediction
    }


    # ========================================================
    # RESULT METRICS
    # ========================================================

    st.header(
        "📊 Assessment Results"
    )


    r1, r2, r3, r4 = st.columns(4)


    r1.metric(
        "Risk Level",
        level
    )


    r2.metric(
        "Concern Index",
        f"{composite:.1f}/100"
    )


    r3.metric(
        "ML Confidence",
        f"{ml_confidence:.1f}%"
    )


    r4.metric(
        "Predicted Trigger",
        prediction
        if prediction
        else "N/A"
    )


    st.progress(
        int(composite)
    )


    # ========================================================
    # WARNING
    # ========================================================

    if level == "CRITICAL":

        st.error(
            "🚨 CRITICAL LANDSLIDE WARNING\n\n"
            f"{risk_message(level)}\n\n"
            f"Location: {city}, {district}, {state}\n"
            f"Concern Index: {composite:.1f}/100"
        )


    elif level == "HIGH":

        st.warning(
            "⚠️ HIGH LANDSLIDE WARNING\n\n"
            f"{risk_message(level)}\n\n"
            f"Location: {city}, {district}, {state}\n"
            f"Concern Index: {composite:.1f}/100"
        )


    elif level == "MODERATE":

        st.info(
            "🟠 MODERATE WATCH\n\n"
            f"{risk_message(level)}\n\n"
            f"Concern Index: {composite:.1f}/100"
        )


    else:

        st.success(
            "🟢 LOW RISK\n\n"
            f"{risk_message(level)}\n\n"
            f"Concern Index: {composite:.1f}/100"
        )


    # ========================================================
    # AUTOMATIC SMS
    # ========================================================

    st.subheader(
        "📱 Automatic SMS Alert"
    )


    if level in [
        "HIGH",
        "CRITICAL"
    ]:

        try:

            sms_configured = all(
                key in st.secrets
                for key in [
                    "TWILIO_ACCOUNT_SID",
                    "TWILIO_AUTH_TOKEN",
                    "TWILIO_PHONE",
                    "RECEIVER_PHONE"
                ]
            )

        except Exception:

            sms_configured = False


        if not sms_configured:

            st.error(
                "⚠️ HIGH/CRITICAL risk detected, "
                "but Twilio SMS is not configured."
            )


        elif analysis_key in st.session_state.sms_sent_keys:

            st.success(
                "✅ SMS already sent for this assessment."
            )


        else:

            # ------------------------------------------------
            # AUTOMATIC SMS SENDING
            # ------------------------------------------------

            with st.spinner(
                "Sending automatic warning SMS..."
            ):

                sms_success, sms_result = (
                    send_sms_alert(
                        level,
                        composite,
                        state,
                        district,
                        city,
                        prediction
                    )
                )


            if sms_success:

                st.success(
                    "📱 ✅ AUTOMATIC SMS ALERT SENT"
                )

                st.session_state.sms_sent_keys.add(
                    analysis_key
                )


                st.session_state.alert_history.append(
                    {

                        "Time":
                            datetime.now()
                            .strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "Location":
                            f"{city}, {district}, {state}",

                        "Risk":
                            level,

                        "Score":
                            round(
                                composite,
                                1
                            ),

                        "Trigger":
                            prediction,

                        "Status":
                            "AUTOMATIC SMS SENT"
                    }
                )


                st.caption(
                    "Twilio Message ID: "
                    + str(sms_result)
                )


            else:

                st.error(
                    "❌ AUTOMATIC SMS FAILED"
                )

                st.code(
                    str(sms_result)
                )


    else:

        st.info(
            f"📱 No SMS sent. Current risk level: {level}."
        )


    # ========================================================
    # MULTI-FACTOR SIGNALS
    # ========================================================

    st.subheader(
        "🔎 Multi-Factor Warning Signals"
    )


    signal_df = pd.DataFrame(
        {

            "Risk Factor": [

                "Historical ML",

                "Rainfall",

                "Seismic Activity",

                "Storm / Wind",

                "Snow / Snowmelt"
            ],

            "Score": [

                round(
                    ml_confidence,
                    2
                ),

                round(
                    rain_score,
                    2
                ),

                round(
                    seismic_score,
                    2
                ),

                round(
                    storm_score,
                    2
                ),

                round(
                    snow_score,
                    2
                )
            ],

            "Status": [

                signal_status(
                    ml_confidence
                ),

                signal_status(
                    rain_score
                ),

                signal_status(
                    seismic_score
                ),

                signal_status(
                    storm_score
                ),

                signal_status(
                    snow_score
                )
            ]
        }
    )


    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RISK FACTOR CHART
    # ========================================================

    st.subheader(
        "📈 Risk Factor Comparison"
    )


    st.bar_chart(
        signal_df.set_index(
            "Risk Factor"
        )[["Score"]],
        height=350
    )


    # ========================================================
    # ENVIRONMENTAL CHART
    # ========================================================

    if weather:

        st.subheader(
            "🌧️ Environmental Profile"
        )


        environmental_df = pd.DataFrame(
            {

                "Factor": [

                    "Current Rain",

                    "Rain - 72h",

                    "Snowfall - 72h",

                    "Maximum Wind"
                ],

                "Value": [

                    weather[
                        "rain"
                    ],

                    weather[
                        "rain_72h"
                    ],

                    weather[
                        "snowfall_72h"
                    ],

                    weather[
                        "max_wind"
                    ]
                ]
            }
        )


        st.bar_chart(
            environmental_df.set_index(
                "Factor"
            ),
            height=300
        )


    # ========================================================
    # ML PROBABILITIES
    # ========================================================

    if probabilities is not None:

        st.subheader(
            "🤖 AI Trigger Probability Distribution"
        )


        classes = get_model_classes()


        if len(classes) == len(
            probabilities
        ):

            probability_df = pd.DataFrame(
                {

                    "Trigger":
                        classes,

                    "Probability":
                        [
                            round(
                                float(
                                    probability
                                ) * 100,
                                2
                            )

                            for probability
                            in probabilities
                        ]
                }
            ).sort_values(
                "Probability",
                ascending=False
            )


            st.bar_chart(
                probability_df.set_index(
                    "Trigger"
                )
            )


            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


    # ========================================================
    # MAP
    # ========================================================

    st.subheader(
        "🗺️ Monitoring Location"
    )


    map_df = pd.DataFrame(
        {
            "latitude":
                [latitude],

            "longitude":
                [longitude]
        }
    )


    st.map(
        map_df,
        latitude="latitude",
        longitude="longitude",
        zoom=8
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    st.subheader(
        "🧠 AI Explanation"
    )


    st.info(
        f"""
Historical ML trigger:
**{prediction}**

Highest ML confidence:
**{ml_confidence:.1f}%**

Current multi-factor Concern Index:
**{composite:.1f}/100**

Current warning level:
**{level}**
"""
    )


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader(
        "🛡️ Recommended Action"
    )


    if level == "CRITICAL":

        st.error(
            "Immediately increase monitoring and follow "
            "official emergency guidance."
        )

    elif level == "HIGH":

        st.warning(
            "Maintain close monitoring and review rainfall, "
            "drainage, slope and seismic conditions."
        )

    elif level == "MODERATE":

        st.info(
            "Continue monitoring environmental and slope conditions."
        )

    else:

        st.success(
            "No strong combined warning signal was detected."
        )


    # ========================================================
    # ASSESSMENT REPORT
    # ========================================================

    st.subheader(
        "📄 Assessment Report"
    )


    report = {

        "system":
            "LandslideShield AI",

        "timestamp":
            datetime.now().isoformat(),

        "assessment_date":
            str(
                analysis_date
            ),

        "location": {

            "state":
                state,

            "district":
                district,

            "city":
                city,

            "latitude":
                latitude,

            "longitude":
                longitude
        },

        "machine_learning": {

            "predicted_trigger":
                prediction,

            "confidence_percent":
                round(
                    ml_confidence,
                    2
                )
        },

        "risk_assessment": {

            "concern_index":
                round(
                    composite,
                    2
                ),

            "risk_level":
                level
        },

        "warning_signals": {

            "rainfall":
                round(
                    rain_score,
                    2
                ),

            "seismic":
                round(
                    seismic_score,
                    2
                ),

            "storm_wind":
                round(
                    storm_score,
                    2
                ),

            "snow_snowmelt":
                round(
                    snow_score,
                    2
                )
        },

        "weather":
            weather,

        "seismic_context":
            earthquake,

        "automatic_sms":
            level in [
                "HIGH",
                "CRITICAL"
            ]
    }


    report_text = json.dumps(
        report,
        indent=4,
        default=str
    )


    st.download_button(
        "⬇️ DOWNLOAD ASSESSMENT REPORT",
        data=report_text,
        file_name="landslideshield_assessment.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================================
# ALERT HISTORY
# ============================================================

st.divider()

st.header(
    "🔔 Alert History"
)


if st.session_state.alert_history:

    history_df = pd.DataFrame(
        st.session_state.alert_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No automatic SMS alerts have been sent "
        "during this session."
    )


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

st.divider()

with st.expander(
    "🤖 Model Intelligence"
):

    if metrics:

        rows = []

        for key, value in metrics.items():

            if isinstance(
                value,
                (int, float)
            ):

                if 0 <= float(value) <= 1:

                    display_value = (
                        f"{float(value) * 100:.2f}%"
                    )

                else:

                    display_value = str(
                        value
                    )

            else:

                display_value = str(
                    value
                )


            rows.append(
                {
                    "Metric":
                        key,

                    "Value":
                        display_value
                }
            )


        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "metrics.json was not found."
        )


    st.write(
        """
The Random Forest model uses historical landslide data
to classify the historical landslide trigger category.
"""
    )

    st.write(
        """
Current weather and seismic information are used as
a separate multi-factor decision-support layer.
"""
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "📘 Methodology"
):

    st.markdown(
        """
### Historical Machine Learning

Historical landslide events are used to train a
Random Forest classification model.

### Model Features

- Latitude
- Longitude
- Year
- Month
- Day
- Day of year
- Season
- State
- Month sine
- Month cosine

### Live Environmental Intelligence

The system retrieves:

- Temperature
- Humidity
- Current rainfall
- 72-hour rainfall
- Snowfall
- Wind

### Seismic Context

Recent earthquake information around the selected
location is used as an additional contextual signal.

### Multi-Factor Concern Index

The current prototype combines:

- ML confidence: 45%
- Rainfall: 25%
- Seismic activity: 15%
- Storm / wind: 10%
- Snow / snowmelt: 5%

### Warning Levels

- LOW: 0–44
- MODERATE: 45–69
- HIGH: 70–84
- CRITICAL: 85–100

### Automatic SMS

HIGH and CRITICAL assessments automatically
send an SMS through Twilio when valid credentials
are configured.

The system prevents the same assessment from
sending repeatedly during the current session.

### Important

The Concern Index is a research/decision-support
score. It is not a calibrated probability that
a landslide is occurring.

This academic prototype should not replace official
disaster-management warnings.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "LandslideShield AI | "
    "AI-Based Multi-Factor Landslide Risk Assessment "
    "& Early Warning Prototype"
)

st.caption(
    "⚠️ Academic / Research Prototype — "
    "Not a replacement for official disaster warnings."
)