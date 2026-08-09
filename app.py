import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Prediction Arena",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background: #0b1020; }
        .block-container { padding-top: 1.2rem; }
        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #111936 0%, #172554 55%, #312e81 100%);
            border: 1px solid rgba(255,255,255,.12);
            margin-bottom: 1rem;
        }
        .hero h1 { margin: 0; font-size: 2.3rem; }
        .hero p { color: #cbd5e1; margin-top: .5rem; }
        .metric-card {
            padding: 1rem;
            border-radius: 14px;
            background: #111827;
            border: 1px solid #263244;
            min-height: 115px;
        }
        .metric-label { color: #94a3b8; font-size: .85rem; }
        .metric-value { font-size: 1.55rem; font-weight: 700; margin-top: .35rem; }
        .badge {
            display: inline-block;
            padding: .35rem .7rem;
            border-radius: 999px;
            background: #1e293b;
            border: 1px solid #334155;
            margin-right: .4rem;
            font-size: .82rem;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 12px;
            background: #3b2f0b;
            border: 1px solid #8a6d1d;
            color: #fde68a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

SEED = 42
TARGET = "target_sales"
FEATURES = [
    "sales",
    "market_indicator_1",
    "market_indicator_2",
    "gdp_growth",
    "unemployment_rate",
    "inflation_rate",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    candidates = [
        Path("simulated_financial_forecasting_data.csv"),
        Path("data/simulated_financial_forecasting_data.csv"),
        Path("/content/simulated_financial_forecasting_data.csv"),
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path)

    return None


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
@st.cache_resource
def train_models(df):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    alpha_grid = {"model__alpha": np.logspace(-3, 3, 25)}

    models = {
        "Linear Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge (tuned)": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(random_state=SEED)),
                ]
            ),
            param_grid=alpha_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "Lasso (tuned)": GridSearchCV(
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Lasso(random_state=SEED, max_iter=20_000)),
                ]
            ),
            param_grid=alpha_grid,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        ),
        "Random Forest": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=400,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    fitted = {}
    rows = []

    mean_model = DummyRegressor(strategy="mean").fit(X_train, y_train)
    mean_pred = mean_model.predict(X_test)
    rows.append(
        {
            "Model": "Baseline: mean",
            "MAE": mean_absolute_error(y_test, mean_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, mean_pred)),
            "R2": r2_score(y_test, mean_pred),
            "MAPE_%": mean_absolute_percentage_error(y_test, mean_pred) * 100,
            "CV_RMSE": np.nan,
            "CV_RMSE_std": np.nan,
        }
    )

    sales_model = Pipeline(
        [("scale", StandardScaler()), ("lr", LinearRegression())]
    ).fit(X_train[["sales"]], y_train)
    sales_pred = sales_model.predict(X_test[["sales"]])
    rows.append(
        {
            "Model": "Baseline: sales-only LR",
            "MAE": mean_absolute_error(y_test, sales_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, sales_pred)),
            "R2": r2_score(y_test, sales_pred),
            "MAPE_%": mean_absolute_percentage_error(y_test, sales_pred) * 100,
            "CV_RMSE": np.nan,
            "CV_RMSE_std": np.nan,
        }
    )

    for name, estimator in models.items():
        estimator.fit(X_train, y_train)
        fitted[name] = estimator

        pred = estimator.predict(X_test)
        cv_rmse = -cross_val_score(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
                "R2": r2_score(y_test, pred),
                "MAPE_%": mean_absolute_percentage_error(y_test, pred) * 100,
                "CV_RMSE": cv_rmse.mean(),
                "CV_RMSE_std": cv_rmse.std(),
            }
        )

    comparison = (
        pd.DataFrame(rows)
        .set_index("Model")
        .sort_values("RMSE")
    )

    best_name = comparison.index[0]
    best_model = fitted.get(best_name, sales_model)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "cv": cv,
        "models": fitted,
        "sales_model": sales_model,
        "comparison": comparison,
        "best_name": best_name,
        "best_model": best_model,
    }


@st.cache_data
def calculate_diagnostics(df, model_bundle, model_name):
    model = model_bundle["models"].get(model_name)

    if model is None:
        model = model_bundle["sales_model"]

    X_train = model_bundle["X_train"]
    y_train = model_bundle["y_train"]
    X_test = model_bundle["X_test"]
    y_test = model_bundle["y_test"]

    test_pred = model.predict(X_test)
    residuals = y_test.values - test_pred

    cv_preds = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=model_bundle["cv"],
        n_jobs=-1,
    )
    cv_residuals = y_train.values - cv_preds
    lower_q, upper_q = np.percentile(cv_residuals, [5, 95])

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=SEED,
        scoring="r2",
        n_jobs=-1,
    )

    importance = (
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Importance": perm.importances_mean,
                "Std": perm.importances_std,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "test_pred": test_pred,
        "residuals": residuals,
        "cv_residuals": cv_residuals,
        "lower_q": float(lower_q),
        "upper_q": float(upper_q),
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>📈 Financial Prediction Arena</h1>
        <p>
            Interactive analytics, model benchmarking, diagnostics and
            conditional target-sales prediction.
        </p>
        <span class="badge">🎯 Prediction Lab</span>
        <span class="badge">📊 Multi-Chart Analytics</span>
        <span class="badge">🏆 Model Arena</span>
        <span class="badge">🎮 Gamified Insights</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload forecasting CSV",
    type=["csv"],
    help="Upload a CSV containing the six predictors and target_sales.",
)

df = load_data(uploaded_file)

if df is None:
    st.error(
        "CSV not found. Place `simulated_financial_forecasting_data.csv` "
        "next to this Streamlit app or upload it from the sidebar."
    )
    st.stop()

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

df = df[FEATURES + [TARGET]].copy()

bundle = train_models(df)

model_options = list(bundle["models"].keys())
default_index = (
    model_options.index(bundle["best_name"])
    if bundle["best_name"] in model_options
    else 0
)

selected_model = st.sidebar.selectbox(
    "Prediction model",
    model_options,
    index=default_index,
)

st.sidebar.caption(
    "The notebook's strongest signal is `sales`; the five macro variables "
    "contribute little on this simulated dataset."
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
comparison = bundle["comparison"]
best_row = comparison.loc[bundle["best_name"]]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Rows</div>'
        f'<div class="metric-value">{len(df):,}</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best Model</div>'
        f'<div class="metric-value">{bundle["best_name"]}</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best Test R²</div>'
        f'<div class="metric-value">{best_row["R2"]:.4f}</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">Best Test RMSE</div>'
        f'<div class="metric-value">{best_row["RMSE"]:,.2f}</div></div>',
        unsafe_allow_html=True,
    )

st.info(
    "Important: the supplied dataset has no date/time column. This application "
    "therefore performs conditional regression (`target_sales` given input features), "
    "not a genuine future time-series forecast."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(
    [
        "🏠 Overview",
        "📊 Exploratory Analytics",
        "🏆 Model Arena",
        "🔬 Diagnostics",
        "🎯 Prediction Lab",
        "🎮 Gamification",
        "📋 Data Explorer",
    ]
)

# ===========================================================================
# OVERVIEW
# ===========================================================================
with tabs[0]:
    st.subheader("Executive Overview")

    a, b = st.columns(2)

    with a:
        fig = px.histogram(
            df,
            x=TARGET,
            nbins=35,
            title="Target Sales Distribution",
            marginal="box",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with b:
        corr = df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()

        fig = px.bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            title="Feature Correlation with Target Sales",
            labels={"x": "Correlation", "y": "Feature"},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    sales_corr = df["sales"].corr(df[TARGET])

    st.markdown(
        f"""
        ### 🔎 Key Finding

        - `sales` correlation with `target_sales`: **{sales_corr:.4f}**
        - The relationship is strongly linear in the supplied simulated data.
        - The five macro indicators show comparatively weak relationships.
        - The notebook recommends the **sales-only model** as the practical production choice.
        """
    )

# ===========================================================================
# EDA
# ===========================================================================
with tabs[1]:
    st.subheader("📊 Exploratory Data Analysis")

    chart_type = st.selectbox(
        "Choose visualization",
        [
            "Scatter Matrix",
            "Correlation Heatmap",
            "Distribution",
            "Box Plot",
            "Violin Plot",
            "Feature vs Target",
        ],
    )

    if chart_type == "Scatter Matrix":
        fig = px.scatter_matrix(
            df.sample(min(500, len(df)), random_state=SEED),
            dimensions=FEATURES,
            color=TARGET,
            title="Interactive Scatter Matrix",
        )
        fig.update_layout(height=850)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Correlation Heatmap":
        corr = df.corr(numeric_only=True)

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                text=np.round(corr.values, 3),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(title="Correlation Matrix", height=650)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Distribution":
        col = st.selectbox("Column", FEATURES + [TARGET])
        fig = px.histogram(
            df,
            x=col,
            nbins=40,
            marginal="box",
            title=f"Distribution — {col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        fig = px.box(
            df,
            y=FEATURES + [TARGET],
            title="Box Plot — All Numeric Variables",
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Violin Plot":
        col = st.selectbox("Column", FEATURES + [TARGET])
        fig = px.violin(
            df,
            y=col,
            box=True,
            points="all",
            title=f"Violin Plot — {col}",
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        feature = st.selectbox("Feature", FEATURES)

        fig = px.scatter(
            df,
            x=feature,
            y=TARGET,
            trendline="ols",
            title=f"{feature} vs {TARGET}",
            hover_data=FEATURES,
        )
        st.plotly_chart(fig, use_container_width=True)

# ===========================================================================
# MODEL ARENA
# ===========================================================================
with tabs[2]:
    st.subheader("🏆 Model Arena")

    display_comparison = comparison.copy()
    display_comparison["R2"] = display_comparison["R2"].round(4)
    display_comparison["MAE"] = display_comparison["MAE"].round(2)
    display_comparison["RMSE"] = display_comparison["RMSE"].round(2)
    display_comparison["MAPE_%"] = display_comparison["MAPE_%"].round(2)
    display_comparison["CV_RMSE"] = display_comparison["CV_RMSE"].round(2)
    display_comparison["CV_RMSE_std"] = display_comparison["CV_RMSE_std"].round(2)

    st.dataframe(display_comparison, use_container_width=True)

    fig = px.bar(
        comparison.reset_index(),
        x="Model",
        y="RMSE",
        title="RMSE Leaderboard — Lower is Better",
        text_auto=".2f",
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        comparison.reset_index(),
        x="Model",
        y="R2",
        title="R² Leaderboard — Higher is Better",
        text_auto=".4f",
    )
    fig.update_layout(xaxis_tickangle=-25)
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"🏆 Current winner by test RMSE: **{bundle['best_name']}**"
    )

# ===========================================================================
# DIAGNOSTICS
# ===========================================================================
with tabs[3]:
    st.subheader(f"🔬 Diagnostics — {selected_model}")

    diag = calculate_diagnostics(df, bundle, selected_model)
    y_test = bundle["y_test"]

    d1, d2 = st.columns(2)

    with d1:
        fig = px.scatter(
            x=y_test,
            y=diag["test_pred"],
            labels={"x": "Actual", "y": "Predicted"},
            title="Predicted vs Actual",
        )
        lo = min(y_test.min(), diag["test_pred"].min())
        hi = max(y_test.max(), diag["test_pred"].max())
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash"},
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with d2:
        fig = px.scatter(
            x=diag["test_pred"],
            y=diag["residuals"],
            labels={"x": "Predicted", "y": "Residual"},
            title="Residuals vs Predicted",
        )
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    d3, d4 = st.columns(2)

    with d3:
        fig = px.histogram(
            x=diag["residuals"],
            nbins=30,
            title="Residual Distribution",
            labels={"x": "Residual"},
        )
        fig.add_vline(x=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

    with d4:
        fig = px.bar(
            diag["importance"].sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            error_x="Std",
            title="Permutation Feature Importance",
        )
        st.plotly_chart(fig, use_container_width=True)

    residual_std = np.std(diag["residuals"])
    st.metric("Residual standard deviation", f"{residual_std:,.2f}")

# ===========================================================================
# PREDICTION LAB
# ===========================================================================
with tabs[4]:
    st.subheader("🎯 Prediction Lab")

    st.markdown(
        "Enter a scenario and press **Predict Target Sales**. "
        "The 90% interval is based on cross-validated training residuals."
    )

    means = df[FEATURES].mean()
    mins = df[FEATURES].min()
    maxs = df[FEATURES].max()

    with st.form("prediction_form"):
        p1, p2, p3 = st.columns(3)

        with p1:
            sales = st.number_input(
                "Sales",
                min_value=float(mins["sales"]),
                max_value=float(maxs["sales"]),
                value=float(means["sales"]),
                step=100.0,
            )
            market_1 = st.number_input(
                "Market Indicator 1",
                min_value=float(mins["market_indicator_1"]),
                max_value=float(maxs["market_indicator_1"]),
                value=float(means["market_indicator_1"]),
                step=1.0,
            )

        with p2:
            market_2 = st.number_input(
                "Market Indicator 2",
                min_value=float(mins["market_indicator_2"]),
                max_value=float(maxs["market_indicator_2"]),
                value=float(means["market_indicator_2"]),
                step=0.5,
            )
            gdp = st.number_input(
                "GDP Growth",
                min_value=float(mins["gdp_growth"]),
                max_value=float(maxs["gdp_growth"]),
                value=float(means["gdp_growth"]),
                step=0.1,
            )

        with p3:
            unemployment = st.number_input(
                "Unemployment Rate",
                min_value=float(mins["unemployment_rate"]),
                max_value=float(maxs["unemployment_rate"]),
                value=float(means["unemployment_rate"]),
                step=0.1,
            )
            inflation = st.number_input(
                "Inflation Rate",
                min_value=float(mins["inflation_rate"]),
                max_value=float(maxs["inflation_rate"]),
                value=float(means["inflation_rate"]),
                step=0.1,
            )

        predict_clicked = st.form_submit_button(
            "🚀 Predict Target Sales",
            use_container_width=True,
            type="primary",
        )

    if predict_clicked:
        input_df = pd.DataFrame(
            [
                {
                    "sales": sales,
                    "market_indicator_1": market_1,
                    "market_indicator_2": market_2,
                    "gdp_growth": gdp,
                    "unemployment_rate": unemployment,
                    "inflation_rate": inflation,
                }
            ]
        )

        model = bundle["models"].get(selected_model, bundle["sales_model"])
        point = float(model.predict(input_df[FEATURES])[0])

        diag = calculate_diagnostics(df, bundle, selected_model)
        lower = point + diag["lower_q"]
        upper = point + diag["upper_q"]

        st.session_state["last_prediction"] = {
            "model": selected_model,
            "point": point,
            "lower": lower,
            "upper": upper,
            "sales": sales,
        }

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Predicted Target Sales", f"{result['point']:,.2f}")

        with r2:
            st.metric("90% Lower Bound", f"{result['lower']:,.2f}")

        with r3:
            st.metric("90% Upper Bound", f"{result['upper']:,.2f}")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=result["point"],
                title={"text": "Predicted Target Sales"},
                gauge={
                    "axis": {
                        "range": [
                            float(df[TARGET].min()),
                            float(df[TARGET].max()),
                        ]
                    }
                },
            )
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.success(
            f"Prediction generated using **{result['model']}**. "
            f"90% residual interval: **{result['lower']:,.2f} – "
            f"{result['upper']:,.2f}**."
        )

# ===========================================================================
# GAMIFICATION
# ===========================================================================
with tabs[5]:
    st.subheader("🎮 Prediction Challenge")

    st.markdown(
        """
        Use this section as a lightweight analytics game: change `sales`,
        make a prediction, and see how your scenario moves relative to the
        historical target distribution.
        """
    )

    sales_value = st.slider(
        "🎚️ Adjust Sales Scenario",
        min_value=float(df["sales"].min()),
        max_value=float(df["sales"].max()),
        value=float(df["sales"].median()),
        step=50.0,
    )

    percentile = float((df["sales"] <= sales_value).mean() * 100)

    if percentile < 25:
        badge = "🟢 Conservative Scenario"
        points = 20
    elif percentile < 50:
        badge = "🔵 Lower-Mid Scenario"
        points = 40
    elif percentile < 75:
        badge = "🟡 Upper-Mid Scenario"
        points = 70
    else:
        badge = "🔴 High-Intensity Scenario"
        points = 100

    g1, g2, g3 = st.columns(3)

    with g1:
        st.metric("Scenario Percentile", f"{percentile:.1f}%")

    with g2:
        st.metric("Challenge Score", f"{points}/100")

    with g3:
        st.metric("Scenario Badge", badge)

    progress = min(points / 100, 1.0)
    st.progress(progress)

    st.caption(
        "Score is a visualization mechanic, not a model-confidence score."
    )

    target_percentile = float(
        (df[TARGET] <= df[TARGET].median()).mean() * 100
    )

    st.markdown(
        f"""
        ### 🏅 Achievement System

        - **Explorer** — inspect at least 3 chart types.
        - **Model Challenger** — compare all model RMSE values.
        - **Diagnostic Detective** — inspect residuals and feature importance.
        - **Prediction Pilot** — generate a scenario prediction.
        - **Insight Master** — identify the dominant `sales` signal.

        Current scenario: **{badge}**
        """
    )

# ===========================================================================
# DATA EXPLORER
# ===========================================================================
with tabs[6]:
    st.subheader("📋 Data Explorer")

    search_term = st.text_input(
        "Filter rows by value",
        placeholder="Example: 5000",
    )

    if search_term:
        numeric_term = pd.to_numeric(search_term, errors="coerce")

        if not np.isnan(numeric_term):
            mask = np.isclose(
                df[FEATURES + [TARGET]].values,
                numeric_term,
                atol=0.5,
            ).any(axis=1)
            display_df = df.loc[mask]
        else:
            display_df = df.iloc[0:0]
    else:
        display_df = df

    st.dataframe(display_df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Dataset",
        data=csv_bytes,
        file_name="financial_forecasting_data.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Financial Prediction Arena • Built with Streamlit + Plotly + scikit-learn • "
    "Use real time-ordered data before describing this as a true forecasting system."
)
