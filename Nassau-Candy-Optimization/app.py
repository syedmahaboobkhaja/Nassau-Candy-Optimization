import os
import warnings
 
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
 
warnings.filterwarnings("ignore")
 
# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy - Factory Optimization",
    page_icon="🍬",
    layout="wide",
)
 
LOGISTICS_COST_PER_KM_PER_UNIT = 0.02
 
FACTORY_COORDS = pd.DataFrame({
    "Factory": [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory",
    ],
    "Latitude": [32.881893, 32.076176, 48.119140, 41.446333, 35.117500],
    "Longitude": [-111.768036, -81.088371, -96.181150, -90.565487, -89.971107],
})
 
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}
 
REGION_CENTERS = {
    "Pacific": (36.7783, -119.4179),
    "Atlantic": (40.7128, -74.0060),
    "Interior": (41.8781, -87.6298),
    "Gulf": (29.7604, -95.3698),
}
 
MODEL_FEATURES = [
    "Product_Name", "Division", "Region", "Ship_Mode",
    "Units", "Sales", "Cost", "Gross_Profit", "Profit_Margin",
    "Shipping_Distance_KM", "Order_Month", "Order_DayOfWeek",
]
TARGET = "Shipping_Lead_Time"
 
CATEGORICAL_FEATURES = ["Product_Name", "Division", "Region", "Ship_Mode"]
NUMERICAL_FEATURES = [
    "Units", "Sales", "Cost", "Gross_Profit", "Profit_Margin",
    "Shipping_Distance_KM", "Order_Month", "Order_DayOfWeek",
]
 
 
# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c
 
 
def calculate_score(lead_time_reduction, logistics_impact, risk_penalty=0):
    speed_score = lead_time_reduction
    profit_score = -logistics_impact
    return speed_score * 0.6 + profit_score * 0.4 - risk_penalty
 
 
@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes_or_path):
    df = pd.read_csv(file_bytes_or_path)
 
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_").str.replace("/", "_")
    )
 
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Ship_Date"] = pd.to_datetime(df["Ship_Date"], errors="coerce")
    df["Shipping_Lead_Time"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
 
    df = df.dropna(subset=[
        "Order_Date", "Ship_Date", "Product_Name",
        "Sales", "Units", "Gross_Profit", "Cost",
    ])
    df = df[df["Shipping_Lead_Time"] >= 0]
    df = df[df["Units"] > 0]
    df = df[df["Sales"] >= 0]
    df = df.reset_index(drop=True)
 
    df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Gross_Profit"] / df["Sales"], 0)
    df["Sales_Per_Unit"] = np.where(df["Units"] != 0, df["Sales"] / df["Units"], 0)
    df["Cost_Per_Unit"] = np.where(df["Units"] != 0, df["Cost"] / df["Units"], 0)
    df["Profit_Per_Unit"] = np.where(df["Units"] != 0, df["Gross_Profit"] / df["Units"], 0)
 
    df["Order_Month"] = df["Order_Date"].dt.month
    df["Order_DayOfWeek"] = df["Order_Date"].dt.dayofweek
    df["Order_Year"] = df["Order_Date"].dt.year
 
    df["Current_Factory"] = df["Product_Name"].map(PRODUCT_FACTORY)
 
    df["Destination_Latitude"] = df["Region"].map(lambda x: REGION_CENTERS.get(x, (np.nan, np.nan))[0])
    df["Destination_Longitude"] = df["Region"].map(lambda x: REGION_CENTERS.get(x, (np.nan, np.nan))[1])
 
    factory_lat = dict(zip(FACTORY_COORDS["Factory"], FACTORY_COORDS["Latitude"]))
    factory_lon = dict(zip(FACTORY_COORDS["Factory"], FACTORY_COORDS["Longitude"]))
    df["Factory_Latitude"] = df["Current_Factory"].map(factory_lat)
    df["Factory_Longitude"] = df["Current_Factory"].map(factory_lon)
 
    df["Shipping_Distance_KM"] = haversine_distance(
        df["Factory_Latitude"], df["Factory_Longitude"],
        df["Destination_Latitude"], df["Destination_Longitude"],
    )
    df["Shipping_Distance_KM"] = df["Shipping_Distance_KM"].fillna(
        df["Shipping_Distance_KM"].median()
    )
 
    return df
 
 
@st.cache_resource(show_spinner=False)
def train_models(df):
    model_df = df[MODEL_FEATURES + [TARGET]].copy().dropna()
    X = model_df[MODEL_FEATURES]
    y = model_df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
 
    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("scaler", StandardScaler())]), NUMERICAL_FEATURES),
        ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
    ])
 
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
 
    results = []
    trained = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        trained[name] = pipe
 
    model_results = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = model_results.iloc[0]["Model"]
    best_model = trained[best_model_name]
 
    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": best_model.predict(X_test),
    })
 
    return {
        "model_results": model_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "predictions_df": predictions_df,
    }
 
 
@st.cache_data(show_spinner=False)
def build_route_clusters(df):
    route_df = (
        df.groupby(["Region", "Ship_Mode", "Division"])
        .agg(
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
            Avg_Distance=("Shipping_Distance_KM", "mean"),
            Avg_Profit_Margin=("Profit_Margin", "mean"),
            Orders=("Order_ID", "count"),
        )
        .reset_index()
    )
 
    cluster_features = ["Avg_Lead_Time", "Avg_Distance", "Avg_Profit_Margin"]
    scaler = StandardScaler()
    cluster_matrix = scaler.fit_transform(route_df[cluster_features])
    kmeans = KMeans(n_clusters=min(4, len(route_df)), random_state=42, n_init=10)
    route_df["Route_Cluster"] = kmeans.fit_predict(cluster_matrix)
    return route_df
 
 
@st.cache_data(show_spinner=False)
def build_scenarios(df, _best_model, best_rmse, best_r2):
    factory_lat = dict(zip(FACTORY_COORDS["Factory"], FACTORY_COORDS["Latitude"]))
    factory_lon = dict(zip(FACTORY_COORDS["Factory"], FACTORY_COORDS["Longitude"]))
    factory_names = FACTORY_COORDS["Factory"].tolist()
    products = df["Product_Name"].dropna().unique()
 
    rows = []
    for product in products:
        product_data = df[df["Product_Name"] == product]
        if len(product_data) == 0:
            continue
        base_row = product_data.iloc[0]
        for factory in factory_names:
            rows.append({
                "Product_Name": product,
                "Division": base_row["Division"],
                "Region": base_row["Region"],
                "Ship_Mode": base_row["Ship_Mode"],
                "Units": base_row["Units"],
                "Sales": base_row["Sales"],
                "Cost": base_row["Cost"],
                "Gross_Profit": base_row["Gross_Profit"],
                "Profit_Margin": base_row["Profit_Margin"],
                "Order_Month": base_row["Order_Month"],
                "Order_DayOfWeek": base_row["Order_DayOfWeek"],
                "Factory": factory,
                "Factory_Latitude": factory_lat[factory],
                "Factory_Longitude": factory_lon[factory],
                "Destination_Latitude": base_row["Destination_Latitude"],
                "Destination_Longitude": base_row["Destination_Longitude"],
            })
 
    scenario_df = pd.DataFrame(rows)
 
    scenario_df["Shipping_Distance_KM"] = haversine_distance(
        scenario_df["Factory_Latitude"], scenario_df["Factory_Longitude"],
        scenario_df["Destination_Latitude"], scenario_df["Destination_Longitude"],
    )
 
    scenario_X = scenario_df[MODEL_FEATURES].copy()
    scenario_df["Predicted_Lead_Time"] = _best_model.predict(scenario_X).clip(min=0)
 
    scenario_df["Current_Factory"] = scenario_df["Product_Name"].map(PRODUCT_FACTORY)
 
    current_lead_times = (
        scenario_df[scenario_df["Factory"] == scenario_df["Current_Factory"]]
        [["Product_Name", "Predicted_Lead_Time"]]
        .rename(columns={"Predicted_Lead_Time": "Current_Predicted_Lead_Time"})
    )
    scenario_df = scenario_df.merge(current_lead_times, on="Product_Name", how="left")
 
    scenario_df["Lead_Time_Reduction_Days"] = (
        scenario_df["Current_Predicted_Lead_Time"] - scenario_df["Predicted_Lead_Time"]
    )
    scenario_df["Lead_Time_Reduction_Percent"] = np.where(
        scenario_df["Current_Predicted_Lead_Time"] > 0,
        (scenario_df["Lead_Time_Reduction_Days"] / scenario_df["Current_Predicted_Lead_Time"]) * 100,
        0,
    )
 
    current_distances = (
        scenario_df[scenario_df["Factory"] == scenario_df["Current_Factory"]]
        [["Product_Name", "Shipping_Distance_KM"]]
        .rename(columns={"Shipping_Distance_KM": "Current_Distance_KM"})
    )
    scenario_df = scenario_df.merge(current_distances, on="Product_Name", how="left")
    scenario_df["Distance_Change_KM"] = (
        scenario_df["Shipping_Distance_KM"] - scenario_df["Current_Distance_KM"]
    )
 
    scenario_df["Estimated_Logistics_Impact"] = (
        scenario_df["Distance_Change_KM"] * scenario_df["Units"] * LOGISTICS_COST_PER_KM_PER_UNIT
    )
 
    scenario_df["Risk_Level"] = np.select(
        [
            scenario_df["Lead_Time_Reduction_Percent"] < 0,
            scenario_df["Lead_Time_Reduction_Percent"].between(0, 10),
            scenario_df["Lead_Time_Reduction_Percent"] > 10,
        ],
        ["High", "Medium", "Low"],
        default="Medium",
    )
    scenario_df["Risk_Penalty"] = scenario_df["Risk_Level"].map({"Low": 0, "Medium": 5, "High": 15})
 
    scenario_df["Recommendation_Score"] = scenario_df.apply(
        lambda row: calculate_score(
            row["Lead_Time_Reduction_Percent"],
            row["Estimated_Logistics_Impact"],
            row["Risk_Penalty"],
        ),
        axis=1,
    )
 
    confidence_base = max(0, min(100, best_r2 * 100))
    scenario_df["Scenario_Confidence_Score"] = confidence_base
 
    alternative_scenarios = scenario_df[
        scenario_df["Factory"] != scenario_df["Current_Factory"]
    ].copy()
 
    recommendations = alternative_scenarios.sort_values(
        ["Product_Name", "Recommendation_Score"], ascending=[True, False]
    )
 
    best_recommendations = (
        recommendations.sort_values("Recommendation_Score", ascending=False)
        .groupby("Product_Name")
        .head(1)
        .reset_index(drop=True)
    )
    best_recommendations["Scenario_Confidence_Score"] = confidence_base
 
    return scenario_df, recommendations, best_recommendations
 
 
def build_kpis(df, best_recommendations, recommendation_coverage):
    avg_lead_reduction = best_recommendations["Lead_Time_Reduction_Percent"].mean()
    avg_profit_impact = best_recommendations["Estimated_Logistics_Impact"].mean()
    avg_confidence = best_recommendations["Scenario_Confidence_Score"].mean()
 
    return pd.DataFrame({
        "KPI": [
            "Lead Time Reduction (%)",
            "Profit/Logistics Impact",
            "Scenario Confidence Score",
            "Recommendation Coverage (%)",
        ],
        "Value": [avg_lead_reduction, avg_profit_impact, avg_confidence, recommendation_coverage],
    })
 
 
# --------------------------------------------------------------------------
# Sidebar - data input
# --------------------------------------------------------------------------
st.sidebar.title("🍬 Nassau Candy")
st.sidebar.caption("Factory Optimization & Lead Time Prediction")
 
uploaded_file = st.sidebar.file_uploader(
    "Upload 'Nassau Candy Distributor.csv'", type=["csv"]
)
 
default_path = "Nassau Candy Distributor.csv"
data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif os.path.exists(default_path):
    data_source = default_path
 
if data_source is None:
    st.title("🍬 Nassau Candy Distributor - Factory Optimization")
    st.info(
        "Upload the **Nassau Candy Distributor.csv** file in the sidebar to get started."
    )
    st.stop()
 
with st.spinner("Cleaning data and engineering features..."):
    df = load_and_clean(data_source)
 
# --------------------------------------------------------------------------
# Header + tabs
# --------------------------------------------------------------------------
st.title("🍬 Nassau Candy Distributor - Factory Optimization")
st.caption(
    f"{df.shape[0]:,} cleaned orders across {df['Product_Name'].nunique()} products "
    f"and {df['Region'].nunique()} regions."
)
 
tab_overview, tab_models, tab_routes, tab_recs, tab_kpis = st.tabs(
    ["📊 Overview", "🤖 Model Comparison", "🗺️ Route Clusters", "🏭 Factory Recommendations", "📈 KPIs"]
)
 
# --------------------------------------------------------------------------
# Overview tab
# --------------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{df.shape[0]:,}")
    col2.metric("Avg Lead Time (days)", f"{df['Shipping_Lead_Time'].mean():.2f}")
    col3.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col4.metric("Avg Profit Margin", f"{df['Profit_Margin'].mean() * 100:.1f}%")
 
    st.subheader("Shipping Lead Time Distribution")
    fig = px.histogram(df, x="Shipping_Lead_Time", nbins=30, marginal="box")
    st.plotly_chart(fig, use_container_width=True)
 
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Avg Lead Time by Region")
        region_avg = df.groupby("Region")["Shipping_Lead_Time"].mean().reset_index()
        st.plotly_chart(px.bar(region_avg, x="Region", y="Shipping_Lead_Time"), use_container_width=True)
    with col_b:
        st.subheader("Avg Lead Time by Ship Mode")
        mode_avg = df.groupby("Ship_Mode")["Shipping_Lead_Time"].mean().reset_index()
        st.plotly_chart(px.bar(mode_avg, x="Ship_Mode", y="Shipping_Lead_Time"), use_container_width=True)
 
    st.subheader("Top 20 Products by Avg Lead Time")
    product_analysis = (
        df.groupby("Product_Name")
        .agg(
            Orders=("Order_ID", "count"),
            Units=("Units", "sum"),
            Sales=("Sales", "sum"),
            Gross_Profit=("Gross_Profit", "sum"),
            Avg_Lead_Time=("Shipping_Lead_Time", "mean"),
        )
        .sort_values("Avg_Lead_Time", ascending=False)
    )
    st.dataframe(product_analysis.head(20), use_container_width=True)
 
# --------------------------------------------------------------------------
# Model comparison tab
# --------------------------------------------------------------------------
with tab_models:
    with st.spinner("Training and comparing models (Linear Regression, Random Forest, Gradient Boosting)..."):
        model_bundle = train_models(df)
 
    st.subheader("Model Comparison")
    st.dataframe(
        model_bundle["model_results"].style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.3f}"}),
        use_container_width=True,
    )
    st.success(f"Best model: **{model_bundle['best_model_name']}**")
 
    st.subheader(f"Actual vs Predicted Lead Time - {model_bundle['best_model_name']}")
    pred_df = model_bundle["predictions_df"]
    fig2 = px.scatter(pred_df, x="Actual", y="Predicted", opacity=0.6)
    min_v, max_v = pred_df["Actual"].min(), pred_df["Actual"].max()
    fig2.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="red", dash="dash"))
    st.plotly_chart(fig2, use_container_width=True)
 
# --------------------------------------------------------------------------
# Route clusters tab
# --------------------------------------------------------------------------
with tab_routes:
    route_df = build_route_clusters(df)
    st.subheader("Route Performance & Clusters (Region x Ship Mode x Division)")
    st.dataframe(route_df, use_container_width=True)
 
    fig3 = px.scatter(
        route_df, x="Avg_Distance", y="Avg_Lead_Time",
        color=route_df["Route_Cluster"].astype(str),
        size="Orders", hover_data=["Region", "Ship_Mode", "Division"],
        labels={"color": "Cluster"},
    )
    st.plotly_chart(fig3, use_container_width=True)
 
# --------------------------------------------------------------------------
# Factory recommendations tab
# --------------------------------------------------------------------------
with tab_recs:
    best_model_name = model_bundle["best_model_name"]
    best_rmse = model_bundle["model_results"].iloc[0]["RMSE"]
    best_r2 = model_bundle["model_results"].iloc[0]["R2"]
 
    with st.spinner("Scoring every product x factory scenario..."):
        scenario_df, recommendations, best_recommendations = build_scenarios(
            df, model_bundle["best_model"], best_rmse, best_r2
        )
 
    total_products = df["Product_Name"].nunique()
    recommended_products = best_recommendations["Product_Name"].nunique()
    recommendation_coverage = (recommended_products / total_products) * 100
 
    st.subheader("Best Recommended Factory per Product")
    st.dataframe(
        best_recommendations[[
            "Product_Name", "Current_Factory", "Factory", "Predicted_Lead_Time",
            "Lead_Time_Reduction_Days", "Lead_Time_Reduction_Percent",
            "Estimated_Logistics_Impact", "Risk_Level", "Recommendation_Score",
        ]].sort_values("Recommendation_Score", ascending=False),
        use_container_width=True,
    )
 
    st.divider()
    st.subheader("Explore Recommendations by Product")
    product_choice = st.selectbox("Select a product", sorted(df["Product_Name"].dropna().unique()))
    product_recs = recommendations[recommendations["Product_Name"] == product_choice]
    st.dataframe(
        product_recs[[
            "Factory", "Predicted_Lead_Time", "Current_Predicted_Lead_Time",
            "Lead_Time_Reduction_Days", "Lead_Time_Reduction_Percent",
            "Estimated_Logistics_Impact", "Risk_Level", "Recommendation_Score",
        ]],
        use_container_width=True,
    )
 
# --------------------------------------------------------------------------
# KPIs tab
# --------------------------------------------------------------------------
with tab_kpis:
    kpi_df = build_kpis(df, best_recommendations, recommendation_coverage)
    st.subheader("Final KPI Summary")
 
    cols = st.columns(len(kpi_df))
    for c, (_, row) in zip(cols, kpi_df.iterrows()):
        c.metric(row["KPI"], f"{row['Value']:.2f}")
 
    st.dataframe(kpi_df, use_container_width=True)
 
    st.divider()
    st.subheader("Download Project Outputs")
 
    out_dir = "nassau_project_outputs"
    os.makedirs(out_dir, exist_ok=True)
 
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    with col_dl1:
        st.download_button(
            "⬇️ Cleaned Data (CSV)", df.to_csv(index=False).encode("utf-8"),
            "cleaned_nassau_data.csv", "text/csv",
        )
        st.download_button(
            "⬇️ Recommendations (CSV)", best_recommendations.to_csv(index=False).encode("utf-8"),
            "recommendations.csv", "text/csv",
        )
    with col_dl2:
        st.download_button(
            "⬇️ Scenario Data (CSV)", scenario_df.to_csv(index=False).encode("utf-8"),
            "factory_scenarios.csv", "text/csv",
        )
        st.download_button(
            "⬇️ Route Clusters (CSV)", build_route_clusters(df).to_csv(index=False).encode("utf-8"),
            "route_clusters.csv", "text/csv",
        )
    with col_dl3:
        st.download_button(
            "⬇️ Model Results (CSV)", model_bundle["model_results"].to_csv(index=False).encode("utf-8"),
            "model_results.csv", "text/csv",
        )
        st.download_button(
            "⬇️ KPI Summary (CSV)", kpi_df.to_csv(index=False).encode("utf-8"),
            "kpis.csv", "text/csv",
        )
 
    if st.button("💾 Save trained model (.pkl) to disk"):
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model_bundle["best_model"], os.path.join(out_dir, "best_lead_time_model.pkl"))
        joblib.dump(FACTORY_COORDS, os.path.join(out_dir, "factory_coordinates.pkl"))
        st.success(f"Model saved to {out_dir}/best_lead_time_model.pkl")
 

