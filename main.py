
import warnings
import re

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
    "Analyze historical payments indicators, compare forecasting "
    "models, identify unusual trends, and select a working estimate."
)

st.info(
    "Forecasts are estimates, not verified facts. Validate results "
    "against secondary research and market developments."
)


# =========================================================
# SAMPLE DATA — HORIZONTAL FORMAT
# =========================================================

def get_sample_data():
    return pd.DataFrame({
        "Indicator": [
            "Debit cards",
            "Credit cards",
            "POS terminals"
        ],
        "Units": ["000s", "000s", "000s"],
        2020: [215.4463, 120.2, 45.1],
        2021: [279.6480, 128.5, 49.3],
        2022: [281.5387, 140.3, 52.8],
        2023: [310.3870, 151.2, 58.2],
        2024: [312.0795, 165.8, 62.5],
        2025: [345.3122, 180.4, 69.1]
    })


# =========================================================
# IDENTIFY YEAR COLUMNS
# =========================================================

def find_year_columns(columns):
    result = []

    for col in columns:
        match = re.fullmatch(
            r"\s*((?:19|20)\d{2})\s*",
            str(col)
        )

        if match:
            year = int(match.group(1))

            if 1900 <= year <= 2100:
                result.append((col, year))

    return sorted(result, key=lambda item: item[1])


# =========================================================
# VALIDATE TIME SERIES
# =========================================================

def validate_data(df):
    data = df[["Year", "Value"]].copy()

    data["Year"] = pd.to_numeric(
        data["Year"], errors="coerce"
    )
    data["Value"] = pd.to_numeric(
        data["Value"], errors="coerce"
    )

    data = data.dropna().sort_values("Year").reset_index(drop=True)

    if len(data) < 3:
        raise ValueError(
            "At least 3 valid annual observations are required. "
            "Five to seven or more are preferable."
        )

    if not np.all(data["Year"] == np.floor(data["Year"])):
        raise ValueError("Year values must be whole numbers.")

    data["Year"] = data["Year"].astype(int)

    if data["Year"].duplicated().any():
        raise ValueError(
            "Duplicate years detected. Keep one value per year."
        )

    messages = []

    expected_years = set(
        range(
            int(data["Year"].min()),
            int(data["Year"].max()) + 1
        )
    )

    missing_years = sorted(
        expected_years - set(data["Year"])
    )

    if missing_years:
        messages.append(
            f"Missing years: {missing_years}. "
            "Check whether observations are unavailable."
        )

    if (data["Value"] < 0).any():
        messages.append(
            "Negative values found. Verify whether they are valid "
            "for this indicator."
        )

    if (data["Value"] == 0).any():
        messages.append(
            "Zero values found. Some growth calculations may be undefined."
        )

    return data, messages


# =========================================================
# FORECASTING MODELS
# =========================================================

def forecast_naive(y, horizon):
    return np.repeat(float(y.iloc[-1]), horizon)


def forecast_moving_average(y, horizon):
    window = min(3, len(y))
    average = float(y.tail(window).mean())

    return np.repeat(average, horizon)


def forecast_cagr(y, horizon):
    first = float(y.iloc[0])
    last = float(y.iloc[-1])

    if first <= 0 or last <= 0:
        raise ValueError(
            "CAGR requires positive first and last observations."
        )

    rate = (last / first) ** (1 / (len(y) - 1)) - 1

    return np.array([
        last * (1 + rate) ** step
        for step in range(1, horizon + 1)
    ])


def forecast_linear_regression(y, horizon):
    x = np.arange(len(y)).reshape(-1, 1)

    future_x = np.arange(
        len(y), len(y) + horizon
    ).reshape(-1, 1)

    model = LinearRegression().fit(
        x, y.astype(float).to_numpy()
    )

    return model.predict(future_x)


def forecast_holt(y, horizon):
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


def forecast_drift(y, horizon):
    average_change = (
        float(y.iloc[-1]) - float(y.iloc[0])
    ) / (len(y) - 1)

    return np.array([
        float(y.iloc[-1]) + average_change * step
        for step in range(1, horizon + 1)
    ])


def forecast_arima(y, horizon):
    if len(y) < 8:
        raise ValueError(
            "ARIMA is only attempted with at least 8 observations."
        )

    model = ARIMA(
        y.astype(float),
        order=(1, 1, 0)
    ).fit()

    return np.asarray(model.forecast(horizon), dtype=float)


MODELS = {
    "Naive": forecast_naive,
    "Moving average (3)": forecast_moving_average,
    "CAGR": forecast_cagr,
    "Linear regression": forecast_linear_regression,
    "Holt damped trend": forecast_holt,
    "Drift": forecast_drift,
    "ARIMA (optional)": forecast_arima
}


# =========================================================
# GENERATE FORECASTS
# =========================================================

def generate_forecasts(
    data,
    horizon,
    include_arima,
    non_negative
):
    y = data["Value"].astype(float).reset_index(drop=True)

    future_years = np.arange(
        int(data["Year"].max()) + 1,
        int(data["Year"].max()) + horizon + 1
    )

    result = pd.DataFrame({"Year": future_years})
    status = {}

    for name, function in MODELS.items():

        if name == "ARIMA (optional)" and not include_arima:
            continue

        try:
            values = np.asarray(
                function(y, horizon),
                dtype=float
            )

            if len(values) != horizon:
                raise ValueError("Unexpected forecast length.")

            if not np.all(np.isfinite(values)):
                raise ValueError("Invalid forecast values.")

            if non_negative:
                values = np.maximum(values, 0)

            result[name] = values
            status[name] = "Available"

        except Exception as exc:
            status[name] = f"Unavailable: {exc}"

    return result, status


# =========================================================
# HISTORICAL BACKTESTING
# =========================================================

def run_backtesting(
    data,
    include_arima,
    non_negative,
    min_train=4
):
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

                if non_negative:
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
                "MAE": np.mean(absolute_errors),
                "WAPE (%)": (
                    100 * np.sum(absolute_errors) / denominator
                    if denominator else np.nan
                ),
                "Bias": np.mean(errors),
                "Direction accuracy (%)": (
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
    history["Growth (%)"] = (
        history["Value"].pct_change() * 100
    )

    # Historical changes
    for _, row in history.dropna(
        subset=["Growth (%)"]
    ).iterrows():

        growth = float(row["Growth (%)"])

        if abs(growth) >= threshold:
            flags.append({
                "Type": "Historical warning",
                "Year": int(row["Year"]),
                "Model": "Actual",
                "Value": float(row["Value"]),
                "Growth (%)": growth,
                "Reason": (
                    "Large annual change. Check the source and "
                    "market context."
                )
            })

    # Forecast changes, including the latest actual-to-forecast change
    for model in [
        col for col in forecasts.columns if col != "Year"
    ]:

        values = np.concatenate([
            [float(data["Value"].iloc[-1])],
            forecasts[model].to_numpy(dtype=float)
        ])

        for i in range(1, len(values)):
            previous = values[i - 1]
            current = values[i]

            if previous == 0:
                continue

            growth = (current / previous - 1) * 100

            if abs(growth) >= threshold:
                flags.append({
                    "Type": "Forecast warning",
                    "Year": int(data["Year"].max() + i),
                    "Model": model,
                    "Value": float(current),
                    "Growth (%)": float(growth),
                    "Reason": (
                        "Projected change exceeds the selected threshold."
                    )
                })

    return pd.DataFrame(
        flags,
        columns=[
            "Type", "Year", "Model", "Value",
            "Growth (%)", "Reason"
        ]
    )


# =========================================================
# MODEL CHART
# =========================================================

def draw_model_chart(data, forecasts, model, indicator, unit):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["Year"],
        y=data["Value"],
        mode="lines+markers",
        name="Historical actuals"
    ))

    years = (
        [int(data["Year"].max())]
        + forecasts["Year"].astype(int).tolist()
    )

    values = (
        [float(data["Value"].iloc[-1])]
        + forecasts[model].astype(float).tolist()
    )

    fig.add_trace(go.Scatter(
        x=years,
        y=values,
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
        hovermode="x unified"
    )

    return fig


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header("Forecast settings")

uploaded_file = st.sidebar.file_uploader(
    "Upload historical Excel or CSV",
    type=["xlsx", "xls", "csv"]
)

use_sample = st.sidebar.checkbox(
    "Use sample horizontal data",
    value=uploaded_file is None
)

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    [2, 3, 5],
    index=2
)

include_arima = st.sidebar.checkbox(
    "Include optional ARIMA",
    value=False
)

non_negative = st.sidebar.checkbox(
    "Prevent negative forecasts",
    value=True
)

threshold = st.sidebar.slider(
    "Flag annual growth/decline above (%)",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

indicator = st.sidebar.text_input(
    "Indicator name",
    "Debit cards"
)

unit = st.sidebar.text_input(
    "Measurement unit",
    "000s"
)


# =========================================================
# LOAD HORIZONTAL EXCEL DATA
# =========================================================

try:
    if uploaded_file is not None:

        if uploaded_file.name.lower().endswith(".csv"):
            raw_data = pd.read_csv(uploaded_file)
        else:
            raw_data = pd.read_excel(uploaded_file)

    elif use_sample:
        raw_data = get_sample_data()

    else:
        st.info("Upload a file or enable sample data.")
        st.stop()

    raw_data.columns = [
        str(col).strip() for col in raw_data.columns
    ]

    st.subheader("Uploaded data preview")

    st.dataframe(
        raw_data,
        use_container_width=True,
        hide_index=True
    )

    year_pairs = find_year_columns(raw_data.columns)

    # -----------------------------------------------
    # HORIZONTAL FORMAT
    # Indicator | Units | 2020 | 2021 | ...
    # -----------------------------------------------

    if len(year_pairs) >= 3:

        st.success("Horizontal year-column format detected.")

        year_columns = [col for col, year in year_pairs]

        indicator_col = next(
            (
                col for col in raw_data.columns
                if str(col).strip().lower()
                in ["indicator", "indicators", "metric"]
            ),
            None
        )

        unit_col = next(
            (
                col for col in raw_data.columns
                if str(col).strip().lower()
                in ["unit", "units", "measurement unit"]
            ),
            None
        )

        country_col = next(
            (
                col for col in raw_data.columns
                if str(col).strip().lower()
                in ["country", "market", "geography"]
            ),
            None
        )

        working_data = raw_data.copy()

        # Country selector if available
        if country_col is not None:
            countries = (
                working_data[country_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            selected_country = st.selectbox(
                "Select country/market",
                countries
            )

            working_data = working_data[
                working_data[country_col].astype(str)
                == selected_country
            ]

        # Indicator selector
        if indicator_col is not None:

            indicators = (
                working_data[indicator_col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not indicators:
                st.error("No indicators found in the selected data.")
                st.stop()

            selected_indicator = st.selectbox(
                "Select indicator",
                indicators
            )

            matching_rows = working_data[
                working_data[indicator_col].astype(str)
                == selected_indicator
            ]

            indicator = selected_indicator

        else:
            matching_rows = working_data

        if matching_rows.empty:
            st.error("No row matches the selected indicator.")
            st.stop()

        # Handle duplicate matching indicators
        if len(matching_rows) > 1:

            st.warning(
                "Multiple rows match this selection. Choose the correct "
                "row, particularly if the data contain subtypes."
            )

            row_choice = st.selectbox(
                "Select the appropriate row",
                range(len(matching_rows)),
                format_func=lambda i: (
                    f"Row {i + 1}: "
                    + " | ".join(
                        f"{col}={matching_rows.iloc[i][col]}"
                        for col in matching_rows.columns
                        if col not in year_columns
                    )
                )
            )

            selected_row = matching_rows.iloc[row_choice]

        else:
            selected_row = matching_rows.iloc[0]

        # Automatically read units
        if unit_col is not None and pd.notna(selected_row[unit_col]):
            unit = str(selected_row[unit_col])

        # Convert horizontal years into a time series internally
        data = pd.DataFrame({
            "Year": [year for col, year in year_pairs],
            "Value": [
                pd.to_numeric(
                    pd.Series([selected_row[col]]),
                    errors="coerce"
                ).iloc[0]
                for col, year in year_pairs
            ]
        })

        data = data.dropna(subset=["Value"])

    # -----------------------------------------------
    # OPTIONAL LONG FORMAT SUPPORT
    # -----------------------------------------------

    else:
        st.warning(
            "No horizontal year columns were detected. "
            "Check that your headings are years such as 2020, 2021, 2022."
        )

        columns = list(raw_data.columns)

        if len(columns) < 2:
            st.error("The file must contain at least two columns.")
            st.stop()

        year_index = (
            columns.index("Year")
            if "Year" in columns else 0
        )

        value_index = (
            columns.index("Value")
            if "Value" in columns else min(1, len(columns) - 1)
        )

        col1, col2 = st.columns(2)

        with col1:
            year_col = st.selectbox(
                "Select year column",
                columns,
                index=year_index
            )

        with col2:
            value_col = st.selectbox(
                "Select value column",
                columns,
                index=value_index
            )

        if year_col == value_col:
            st.error("Choose different columns for year and value.")
            st.stop()

        data = raw_data[
            [year_col, value_col]
        ].rename(columns={
            year_col: "Year",
            value_col: "Value"
        })

    data, data_warnings = validate_data(data)

except Exception as exc:
    st.error(f"Input error: {exc}")
    st.stop()


# =========================================================
# HISTORICAL ANALYSIS
# =========================================================

for message in data_warnings:
    st.warning(message)

st.header("1. Historical analysis")

history = data.copy()

history["YoY growth (%)"] = (
    history["Value"].pct_change() * 100
)

latest_year = int(data["Year"].max())
latest_value = float(data["Value"].iloc[-1])
latest_growth = history["YoY growth (%)"].iloc[-1]

if (
    data["Value"].iloc[0] > 0
    and latest_value > 0
):
    historical_cagr = (
        (latest_value / float(data["Value"].iloc[0]))
        ** (1 / (len(data) - 1)) - 1
    ) * 100
else:
    historical_cagr = np.nan

c1, c2, c3, c4 = st.columns(4)

c1.metric("Latest year", latest_year)
c2.metric("Latest value", f"{latest_value:,.4f}")

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

historical_chart = go.Figure()

historical_chart.add_trace(go.Scatter(
    x=data["Year"],
    y=data["Value"],
    mode="lines+markers",
    name="Actual values"
))

historical_chart.update_layout(
    title=f"{indicator} — historical trend",
    xaxis_title="Year",
    yaxis_title=unit,
    template="plotly_white"
)

st.plotly_chart(
    historical_chart,
    use_container_width=True
)


# =========================================================
# FORECAST COMPARISON
# =========================================================

st.header("2. Forecast comparison")

forecasts, model_status = generate_forecasts(
    data,
    horizon,
    include_arima,
    non_negative
)

available_models = [
    col for col in forecasts.columns
    if col != "Year"
]

if not available_models:
    st.error("No models generated usable forecasts.")
    st.json(model_status)
    st.stop()

st.dataframe(
    forecasts.round(4),
    use_container_width=True,
    hide_index=True
)

with st.expander("Model status and assumptions"):
    for name, status in model_status.items():
        st.write(f"**{name}:** {status}")


# =========================================================
# INDIVIDUAL MODEL CHARTS
# =========================================================

st.header("3. Individual model charts")

for model_name in available_models:

    with st.expander(
        f"View chart: {model_name}",
        expanded=True
    ):
        st.plotly_chart(
            draw_model_chart(
                data,
                forecasts,
                model_name,
                indicator,
                unit
            ),
            use_container_width=True
        )


# =========================================================
# BACKTESTING
# =========================================================

st.header("4. Historical backtesting")

st.caption(
    "Lower MAE and WAPE generally indicate smaller errors. Positive bias "
    "means overestimation. Rankings are uncertain with limited observations."
)

backtest_results = run_backtesting(
    data,
    include_arima,
    non_negative
)

if backtest_results.empty:
    st.warning("No successful backtest results are available.")
else:
    st.dataframe(
        backtest_results.round(3),
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# ABNORMAL TREND WARNINGS
# =========================================================

st.header("5. Unusual-trend detection")

st.caption(
    f"Flags annual changes of at least {threshold}%. "
    "This threshold is configurable and not universal."
)

anomalies = detect_anomalies(
    data,
    forecasts,
    threshold
)

if anomalies.empty:
    st.success(
        "No changes exceeded the selected threshold. "
        "This does not guarantee that forecasts are correct."
    )
else:
    st.warning(
        "Review these warnings. Large changes may be valid when "
        "supported by source data or known market developments."
    )

    st.dataframe(
        anomalies.round(3),
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
    min_value=-100.0,
    max_value=1000.0,
    value=0.0,
    step=1.0,
    help="Use an adjustment only when justified by research or market context."
)

reason = st.text_input(
    "Rationale for your selection",
    "Reviewed historical trend, backtesting and market context."
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

if non_negative:
    selected_table["Working estimate"] = (
        selected_table["Working estimate"].clip(lower=0)
    )

selected_table["Rationale"] = reason

st.dataframe(
    selected_table.round(4),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "No Excel export is included. The models use the selected indicator's "
    "historical values and do not automatically account for regulation, "
    "inflation, instant-payment adoption or other external market events."
)
