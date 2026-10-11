
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Payments Forecasting Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Payments Indicator Forecasting Engine")
st.write(
    "Compare forecasting methods, evaluate historical accuracy, "
    "identify unusual trends, and choose a working estimate."
)

st.info(
    "Use forecasts as decision-support estimates. Verify them against "
    "secondary research and market developments before adopting them."
)


# =========================================================
# SAMPLE DATA
# =========================================================

def get_sample_data():
    return pd.DataFrame({
        "Year": list(range(2016, 2026)),
        "Value": [
            5.8, 6.1, 6.6, 7.0, 7.4,
            8.2, 8.7, 9.1, 9.65, 10.2
        ]
    })


# =========================================================
# DATA VALIDATION
# =========================================================

def validate_data(df):
    required = {"Year", "Value"}

    if not required.issubset(df.columns):
        raise ValueError(
            "The selected data must contain Year and Value columns."
        )

    data = df[["Year", "Value"]].copy()

    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")

    data = data.dropna().sort_values("Year").reset_index(drop=True)

    if len(data) < 3:
        raise ValueError(
            "At least 3 valid annual observations are required. "
            "Five to seven or more observations are preferable."
        )

    if not np.all(data["Year"] == np.floor(data["Year"])):
        raise ValueError("Year values must be whole numbers.")

    data["Year"] = data["Year"].astype(int)

    if data["Year"].duplicated().any():
        raise ValueError(
            "Duplicate years found. Keep one observation per year."
        )

    messages = []

    expected = set(
        range(int(data["Year"].min()), int(data["Year"].max()) + 1)
    )
    missing = sorted(expected - set(data["Year"]))

    if missing:
        messages.append(
            f"Missing years: {missing}. Check whether data are unavailable."
        )

    if (data["Value"] < 0).any():
        messages.append(
            "Negative values detected. Confirm whether they are valid "
            "for the selected indicator."
        )

    if (data["Value"] == 0).any():
        messages.append(
            "Zero values detected. Some percentage-growth calculations "
            "may be undefined."
        )

    return data, messages


# =========================================================
# FORECASTING MODELS
# =========================================================

def model_naive(y, horizon):
    return np.repeat(float(y.iloc[-1]), horizon)


def model_moving_average(y, horizon):
    window = min(3, len(y))
    level = float(y.tail(window).mean())

    return np.repeat(level, horizon)


def model_cagr(y, horizon):
    first = float(y.iloc[0])
    last = float(y.iloc[-1])

    if first <= 0 or last <= 0:
        raise ValueError("CAGR requires positive first and last values.")

    rate = (last / first) ** (1 / (len(y) - 1)) - 1

    return np.array([
        last * (1 + rate) ** step
        for step in range(1, horizon + 1)
    ])


def model_linear_regression(y, horizon):
    x = np.arange(len(y)).reshape(-1, 1)

    future_x = np.arange(
        len(y), len(y) + horizon
    ).reshape(-1, 1)

    model = LinearRegression().fit(
        x, y.astype(float).to_numpy()
    )

    return model.predict(future_x)


def model_holt(y, horizon):
    if len(y) < 4:
        raise ValueError("Holt requires at least 4 observations.")

    model = ExponentialSmoothing(
        y.astype(float),
        trend="add",
        damped_trend=True,
        seasonal=None,
        initialization_method="estimated"
    ).fit(optimized=True)

    return np.asarray(model.forecast(horizon), dtype=float)


def model_drift(y, horizon):
    if len(y) < 2:
        raise ValueError("Drift requires at least 2 observations.")

    average_change = (
        float(y.iloc[-1]) - float(y.iloc[0])
    ) / (len(y) - 1)

    return np.array([
        float(y.iloc[-1]) + average_change * step
        for step in range(1, horizon + 1)
    ])


def model_arima(y, horizon):
    if len(y) < 8:
        raise ValueError(
            "ARIMA is disabled for fewer than 8 observations."
        )

    model = ARIMA(
        y.astype(float),
        order=(1, 1, 0)
    ).fit()

    return np.asarray(model.forecast(horizon), dtype=float)


MODELS = {
    "Naive": model_naive,
    "Moving average (3)": model_moving_average,
    "CAGR": model_cagr,
    "Linear regression": model_linear_regression,
    "Holt damped trend": model_holt,
    "Drift": model_drift,
    "ARIMA (optional)": model_arima
}


# =========================================================
# GENERATE FORECASTS
# =========================================================

def generate_forecasts(data, horizon, include_arima, non_negative):
    y = data["Value"].astype(float).reset_index(drop=True)

    future_years = np.arange(
        int(data["Year"].max()) + 1,
        int(data["Year"].max()) + horizon + 1
    )

    forecasts = pd.DataFrame({"Year": future_years})
    status = {}

    for name, function in MODELS.items():

        if name == "ARIMA (optional)" and not include_arima:
            continue

        try:
            values = np.asarray(
                function(y, horizon), dtype=float
            )

            if len(values) != horizon:
                raise ValueError("Unexpected forecast length.")

            if not np.all(np.isfinite(values)):
                raise ValueError("Invalid forecast values.")

            if non_negative:
                values = np.maximum(values, 0)

            forecasts[name] = values
            status[name] = "Available"

        except Exception as exc:
            status[name] = f"Unavailable: {exc}"

    return forecasts, status


# =========================================================
# HISTORICAL BACKTESTING
# =========================================================

def run_backtesting(data, include_arima, min_train=4):
    results = []

    for name, function in MODELS.items():

        if name == "ARIMA (optional)" and not include_arima:
            continue

        errors = []
        actuals = []
        direction_matches = []

        for end in range(min_train, len(data)):

            train = (
                data.loc[:end - 1, "Value"]
                .astype(float)
                .reset_index(drop=True)
            )

            actual = float(data.loc[end, "Value"])
            previous = float(data.loc[end - 1, "Value"])

            try:
                predicted = float(
                    np.asarray(function(train, 1))[0]
                )

                if not np.isfinite(predicted):
                    continue

                predicted = max(0, predicted)

                errors.append(predicted - actual)
                actuals.append(actual)

                actual_direction = np.sign(actual - previous)
                predicted_direction = np.sign(predicted - previous)

                direction_matches.append(
                    float(actual_direction == predicted_direction)
                )

            except Exception:
                continue

        if errors:
            absolute_errors = np.abs(errors)
            denominator = np.sum(np.abs(actuals))

            results.append({
                "Model": name,
                "Test observations": len(errors),
                "MAE": float(np.mean(absolute_errors)),
                "WAPE (%)": (
                    float(100 * np.sum(absolute_errors) / denominator)
                    if denominator != 0 else np.nan
                ),
                "Bias": float(np.mean(errors)),
                "Direction accuracy (%)": float(
                    100 * np.mean(direction_matches)
                )
            })

    result = pd.DataFrame(results)

    if not result.empty:
        result = result.sort_values(
            ["WAPE (%)", "MAE"],
            na_position="last"
        ).reset_index(drop=True)

    return result


# =========================================================
# ANOMALY DETECTION
# =========================================================

def detect_anomalies(data, forecasts, threshold):
    flags = []

    history = data.copy()
    history["Growth (%)"] = history["Value"].pct_change() * 100

    for _, row in history.dropna(subset=["Growth (%)"]).iterrows():

        growth = float(row["Growth (%)"])

        if abs(growth) >= threshold:
            flags.append({
                "Type": "Historical warning",
                "Year": int(row["Year"]),
                "Model": "Actual",
                "Value": float(row["Value"]),
                "Growth (%)": growth,
                "Reason": "Large annual change. Verify data and market context."
            })

    for model in [
        col for col in forecasts.columns if col != "Year"
    ]:

        values = forecasts[model].to_numpy(dtype=float)

        # Include the latest actual value to calculate the first
        # forecast year's growth correctly.
        values_with_actual = np.concatenate([
            [float(data["Value"].iloc[-1])],
            values
        ])

        for i in range(1, len(values_with_actual)):

            previous = values_with_actual[i - 1]
            current = values_with_actual[i]

            if previous == 0:
                continue

            growth = (current / previous - 1) * 100

            if abs(growth) >= threshold:
                forecast_year = int(
                    data["Year"].max() + i
                )

                flags.append({
                    "Type": "Forecast warning",
                    "Year": forecast_year,
                    "Model": model,
                    "Value": float(current),
                    "Growth (%)": float(growth),
                    "Reason": "Forecast change exceeds the selected threshold."
                })

    return pd.DataFrame(
        flags,
        columns=[
            "Type", "Year", "Model", "Value",
            "Growth (%)", "Reason"
        ]
    )


# =========================================================
# INDIVIDUAL MODEL CHART
# =========================================================

def draw_model_chart(data, forecasts, model, indicator, unit):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Year"],
        y=data["Value"],
        mode="lines+markers",
        name="Historical actuals"
    ))

    chart_years = [
        int(data["Year"].max())
    ] + forecasts["Year"].astype(int).tolist()

    chart_values = [
        float(data["Value"].iloc[-1])
    ] + forecasts[model].astype(float).tolist()

    fig.add_trace(go.Scatter(
        x=chart_years,
        y=chart_values,
        mode="lines+markers",
        name=f"{model} forecast",
        line=dict(dash="dash")
    ))

    fig.add_vline(
        x=int(data["Year"].max()) + 0.5,
        line_dash="dot",
        annotation_text="Forecast starts"
    )

    fig.update_layout(
        title=f"{indicator} — {model}",
        xaxis_title="Year",
        yaxis_title=unit,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h")
    )

    return fig


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Forecast settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload historical data",
    type=["xlsx", "xls", "csv"]
)

use_sample = st.sidebar.checkbox(
    "Use sample payments data",
    value=uploaded_file is None
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [2, 3, 5],
    index=2
)

include_arima = st.sidebar.checkbox(
    "Include optional ARIMA model",
    value=False
)

non_negative = st.sidebar.checkbox(
    "Prevent negative forecasts",
    value=True,
    help="Recommended for counts, transaction values and other non-negative indicators."
)

threshold = st.sidebar.slider(
    "Flag growth/decline above (%)",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

indicator = st.sidebar.text_input(
    "Indicator name",
    "Debit cards in circulation"
)

unit = st.sidebar.text_input(
    "Measurement unit",
    "Million cards"
)


# =========================================================
# LOAD INPUT DATA
# =========================================================

try:

    if uploaded_file is not None:

        if uploaded_file.name.lower().endswith(".csv"):
            raw_data = pd.read_csv(uploaded_file)
        else:
            raw_data = pd.read_excel(uploaded_file)

        st.subheader("Uploaded data preview")
        st.dataframe(
            raw_data.head(20),
            use_container_width=True,
            hide_index=True
        )

        columns = list(raw_data.columns)

        if len(columns) < 2:
            st.error("Your file must have at least two columns.")
            st.stop()

        default_year_index = (
            columns.index("Year") if "Year" in columns else 0
        )

        default_value_index = (
            columns.index("Value") if "Value" in columns else 1
        )

        col1, col2 = st.columns(2)

        with col1:
            year_column = st.selectbox(
                "Select year column",
                columns,
                index=default_year_index
            )

        with col2:
            value_column = st.selectbox(
                "Select indicator value column",
                columns,
                index=default_value_index
            )

        if year_column == value_column:
            st.error("Choose different columns for Year and Value.")
            st.stop()

        data = raw_data[[year_column, value_column]].rename(
            columns={
                year_column: "Year",
                value_column: "Value"
            }
        )

    elif use_sample:
        data = get_sample_data()

    else:
        st.info("Upload an Excel/CSV file or enable sample data.")
        st.stop()

    data, warnings_list = validate_data(data)

except Exception as exc:
    st.error(f"Input error: {exc}")
    st.stop()


# =========================================================
# DISPLAY DATA WARNINGS
# =========================================================

for message in warnings_list:
    st.warning(message)


# =========================================================
# HISTORICAL ANALYSIS
# =========================================================

st.header("1. Historical data analysis")

history = data.copy()
history["YoY growth (%)"] = history["Value"].pct_change() * 100

latest_value = float(data["Value"].iloc[-1])
latest_year = int(data["Year"].max())
latest_growth = history["YoY growth (%)"].iloc[-1]

if len(data) >= 2 and data["Value"].iloc[0] > 0 and latest_value > 0:
    historical_cagr = (
        (latest_value / float(data["Value"].iloc[0]))
        ** (1 / (len(data) - 1)) - 1
    ) * 100
else:
    historical_cagr = np.nan

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest year", latest_year)
c2.metric("Latest value", f"{latest_value:,.3f}")

c3.metric(
    "Latest YoY growth",
    "N/A" if pd.isna(latest_growth)
    else f"{latest_growth:+.2f}%"
)

c4.metric(
    "Historical CAGR",
    "N/A" if pd.isna(historical_cagr)
    else f"{historical_cagr:+.2f}%"
)

st.dataframe(
    history.round(4),
    use_container_width=True,
    hide_index=True
)

historical_fig = go.Figure()

historical_fig.add_trace(go.Scatter(
    x=data["Year"],
    y=data["Value"],
    mode="lines+markers",
    name="Actual values"
))

historical_fig.update_layout(
    title=f"{indicator} — historical trend",
    xaxis_title="Year",
    yaxis_title=unit,
    template="plotly_white"
)

st.plotly_chart(historical_fig, use_container_width=True)


# =========================================================
# FORECAST TABLE
# =========================================================

st.header("2. Forecast comparison")

forecasts, model_status = generate_forecasts(
    data,
    horizon,
    include_arima,
    non_negative
)

available_models = [
    column for column in forecasts.columns
    if column != "Year"
]

if not available_models:
    st.error("No models generated a valid forecast.")
    st.json(model_status)
    st.stop()

st.dataframe(
    forecasts.round(4),
    use_container_width=True,
    hide_index=True
)

with st.expander("Model availability and notes"):
    for model_name, status in model_status.items():
        st.write(f"**{model_name}:** {status}")


# =========================================================
# SEPARATE CHART FOR EVERY MODEL
# =========================================================

st.header("3. Individual model charts")

for model_name in available_models:

    with st.expander(
        f"View chart: {model_name}",
        expanded=True
    ):
        fig = draw_model_chart(
            data,
            forecasts,
            model_name,
            indicator,
            unit
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# BACKTESTING
# =========================================================

st.header("4. Historical backtesting")

st.caption(
    "MAE and WAPE measure forecast error; lower is generally better. "
    "Positive bias indicates overestimation. Direction accuracy measures "
    "whether the model correctly predicts growth, decline or no change. "
    "Results can be unreliable when few test observations are available."
)

backtest_results = run_backtesting(
    data,
    include_arima
)

if backtest_results.empty:

    st.warning(
        "No models produced enough successful backtest predictions."
    )

else:

    st.dataframe(
        backtest_results.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Do not select a model solely because it ranks first. "
        "Consider its assumptions, chart shape, market context and test count."
    )


# =========================================================
# ABNORMAL TREND WARNINGS
# =========================================================

st.header("5. Unusual-trend detection")

st.caption(
    f"Currently flags annual growth or decline of at least "
    f"{threshold}%. This is a screening threshold, not a universal rule."
)

flags = detect_anomalies(
    data,
    forecasts,
    threshold
)

if flags.empty:

    st.success(
        "No changes exceeded the selected threshold. "
        "This does not guarantee the forecast is correct."
    )

else:

    st.warning(
        "Review these warnings. A large change may be valid if supported "
        "by a market event, regulatory change or source data."
    )

    st.dataframe(
        flags.round(3),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SELECT FINAL WORKING ESTIMATE
# =========================================================

st.header("6. Select your working estimate")

selected_model = st.selectbox(
    "Choose a forecasting model",
    available_models
)

adjustment = st.number_input(
    "Optional analyst adjustment (%)",
    min_value=-100.0 if non_negative else -1000.0,
    max_value=1000.0,
    value=0.0,
    step=1.0,
    help="Use only when justified by secondary research or market knowledge."
)

reason = st.text_input(
    "Rationale for your selection",
    "Reviewed historical trends, backtesting and market context."
)

selected_table = forecasts[
    ["Year", selected_model]
].rename(
    columns={selected_model: "Model forecast"}
).copy()

selected_table["Adjustment (%)"] = adjustment

selected_table["Working estimate"] = (
    selected_table["Model forecast"]
    * (1 + adjustment / 100)
)

selected_table["Rationale"] = reason

st.subheader("Your selected forecast")

st.dataframe(
    selected_table.round(4),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The adjustment changes the working estimate, not the original model "
    "forecast. Forecasts are generated from the supplied history only and "
    "do not automatically account for policy changes, inflation, instant "
    "payment adoption or other external developments."
)

st.success(
    "Forecast comparison complete. Review the model charts, validation "
    "results and any warnings before choosing your final estimate."
)
