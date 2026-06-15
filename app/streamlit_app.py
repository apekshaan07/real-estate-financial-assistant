import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "app"))

import config  # noqa: F401 - loads .env
from config import get_database_backend, get_integration_status
from db import filter_properties, get_portfolio_summary, get_property_financials, initialize_database
from press_release_loader import load_press_releases, search_press_releases
from sec_financials import get_latest_sec_summary, load_sec_financials
from agent.runner_service import ask_assistant
from inference.sagemaker_client import (
    predict_housing_value_cloud_or_local,
    predict_subscription_cloud_or_local,
)
from inference.bedrock_summary import summarize_with_bedrock_or_local
from inference.azure_summary import summarize_with_azure_or_local

st.set_page_config(
    page_title="Financial Assistant | Prologis",
    page_icon="🏢",
    layout="wide",
)

initialize_database()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "adk_session_id" not in st.session_state:
    st.session_state.adk_session_id = None

st.title("Financial Assistant Web Application")
st.caption("AI-driven financial assistant for Prologis — property data, SEC filings, press releases, and ML predictions")

page = st.sidebar.radio(
    "Navigation",
    [
        "Assistant",
        "Dashboard",
        "Property Financials",
        "Press Releases",
        "SEC Financials",
        "ML Predictions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
for name, info in get_integration_status().items():
    icon = "✅" if info["status"] == "ready" else "⚠️"
    st.sidebar.caption(f"{icon} **{name.replace('_', ' ').title()}**: {info['detail']}")

portfolio = get_portfolio_summary()
property_df = get_property_financials()


def render_results_panel(result: dict | None) -> None:
    st.subheader("Results")
    if not result:
        st.info("Ask a question in the chat panel to retrieve financial, property, or press release data.")
        return

    st.caption(f"Source: {result.get('data_type', 'N/A')} | Engine: {result.get('engine', 'local')}")
    st.markdown(result.get("answer", ""))

    if result.get("warning"):
        st.warning(result["warning"])

    dataframe = result.get("dataframe")
    if isinstance(dataframe, pd.DataFrame) and not dataframe.empty:
        st.dataframe(dataframe)
    elif result.get("records"):
        st.dataframe(pd.DataFrame(result["records"]))


if page == "Assistant":
    st.subheader("Conversational Financial Assistant")

    chat_col, results_col = st.columns([1, 1], gap="large")

    with chat_col:
        st.markdown("#### Chat")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input("Ask about revenue, properties, SEC filings, or acquisitions...")
        example_questions = [
            "What was the net income reported last quarter?",
            "Show industrial properties in the Chicago region with revenue details",
            "Did the company announce any acquisitions recently?",
        ]
        st.markdown("**Example questions**")
        for example in example_questions:
            if st.button(example, key=f"example-{example}"):
                st.session_state.pending_question = example

        question = user_question or st.session_state.pop("pending_question", None)
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.spinner("Analyzing question and retrieving data..."):
                response = ask_assistant(question, session_id=st.session_state.adk_session_id)
                st.session_state.adk_session_id = response.get("session_id")
                st.session_state.last_result = response
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response.get("answer", "")}
            )
            st.rerun()

    with results_col:
        render_results_panel(st.session_state.last_result)

elif page == "Dashboard":
    st.subheader("Company Financial Overview")
    st.caption(f"Database backend: **{get_database_backend().upper()}**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Properties", portfolio["total_properties"])
    c2.metric("Total Revenue", f"${portfolio['total_revenue']:,.0f}")
    c3.metric("Total Net Income", f"${portfolio['total_net_income']:,.0f}")
    c4.metric("Total Expenses", f"${portfolio['total_expenses']:,.0f}")

    metro_summary = property_df.groupby("metro_area", as_index=False)["revenue"].sum()
    st.bar_chart(metro_summary.set_index("metro_area"))
    st.dataframe(property_df)

elif page == "Property Financials":
    st.subheader("Property and Financial Data Query")
    metro_options = ["All"] + sorted(property_df["metro_area"].unique().tolist())
    type_options = ["All"] + sorted(property_df["property_type"].unique().tolist())

    c1, c2 = st.columns(2)
    selected_metro = c1.selectbox("Metro Area", metro_options)
    selected_type = c2.selectbox("Property Type", type_options)
    filtered_df = filter_properties(selected_metro, selected_type)

    st.write(f"Showing {len(filtered_df)} matching properties")
    st.dataframe(filtered_df)
    if not filtered_df.empty:
        st.metric("Filtered Revenue", f"${filtered_df['revenue'].sum():,.0f}")
        st.metric("Filtered Net Income", f"${filtered_df['net_income'].sum():,.0f}")

elif page == "Press Releases":
    st.subheader("Company Press Releases and Business Insights")
    press_df = load_press_releases()

    c1, c2 = st.columns([2, 1])
    keyword = c1.text_input("Search press releases", placeholder="acquisition, expansion, quarterly update")
    category_options = ["All"] + sorted(press_df["category"].unique().tolist())
    selected_category = c2.selectbox("Category", category_options)

    filtered_press = search_press_releases(keyword)
    if selected_category != "All":
        filtered_press = filtered_press[filtered_press["category"] == selected_category]

    st.dataframe(filtered_press)

    if filtered_press.empty:
        st.warning("No matching press releases found.")
    else:
        combined_text = " ".join(
            f"{row['title']}. {row['summary']}" for _, row in filtered_press.head(3).iterrows()
        )
        st.markdown("#### Multi-Cloud Summaries")
        bedrock_summary = summarize_with_bedrock_or_local(combined_text)
        azure_summary = summarize_with_azure_or_local(combined_text)
        st.info(f"**Bedrock:** {bedrock_summary}")
        st.success(f"**Azure OpenAI:** {azure_summary}")

elif page == "SEC Financials":
    st.subheader("SEC EDGAR Financial Reports")
    sec_df = load_sec_financials()
    st.dataframe(sec_df)
    st.success(get_latest_sec_summary())

    selected_period = st.selectbox("Select Filing Period", sec_df["period"].tolist())
    selected_row = sec_df[sec_df["period"] == selected_period].iloc[0]
    c1, c2, c3 = st.columns(3)
    if pd.notna(selected_row["revenue"]):
        c1.metric("Revenue", f"${selected_row['revenue']:,.0f}")
        c2.metric("Net Income", f"${selected_row['net_income']:,.0f}")
        c3.metric("Operating Expenses", f"${selected_row['operating_expenses']:,.0f}")

elif page == "ML Predictions":
    st.subheader("Machine Learning Predictions")
    st.markdown(
        "These models are trained locally and can be invoked through SageMaker hosted endpoints "
        "when `SAGEMAKER_REGRESSION_ENDPOINT` and `SAGEMAKER_CLASSIFICATION_ENDPOINT` are configured."
    )

    regression_metrics_path = BASE_DIR / "models" / "housing_regression_metrics.json"
    classification_metrics_path = BASE_DIR / "models" / "bank_classification_metrics.json"
    if regression_metrics_path.exists():
        st.markdown("#### Regression Metrics")
        st.json(json.loads(regression_metrics_path.read_text()))
    if classification_metrics_path.exists():
        st.markdown("#### Classification Metrics")
        st.json(json.loads(classification_metrics_path.read_text()))

    st.divider()
    st.markdown("### Housing Value Regression (Random Forest)")
    c1, c2 = st.columns(2)
    with c1:
        med_inc = st.number_input("Median Income", value=5.0)
        house_age = st.number_input("House Age", value=20.0)
        ave_rooms = st.number_input("Average Rooms", value=6.0)
        ave_bedrooms = st.number_input("Average Bedrooms", value=1.1)
    with c2:
        population = st.number_input("Population", value=1200.0)
        ave_occup = st.number_input("Average Occupancy", value=3.0)
        latitude = st.number_input("Latitude", value=34.05)
        longitude = st.number_input("Longitude", value=-118.25)

    if st.button("Predict Housing Value", type="primary"):
        input_data = {
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrooms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude,
        }
        result = predict_housing_value_cloud_or_local(input_data)
        st.caption(f"Inference source: **{result.get('inference_source', 'unknown')}**")
        if result.get("warning"):
            st.warning(result["warning"])
        st.metric("Predicted Median House Value", f"${result['predicted_value_usd']:,.2f}")
        st.json(result)

    st.divider()
    st.markdown("### Bank Marketing Classification (Logistic Regression)")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Customer Age", value=35, key="age")
        job = st.selectbox(
            "Job",
            [
                "management", "technician", "entrepreneur", "blue-collar",
                "unknown", "retired", "admin.", "services", "self-employed",
                "unemployed", "housemaid", "student",
            ],
        )
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["tertiary", "secondary", "primary", "unknown"])
        default = st.selectbox("Has Credit Default?", ["no", "yes"])
        balance = st.number_input("Account Balance", value=1500)
    with c2:
        housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
        contact = st.selectbox("Contact Type", ["cellular", "unknown", "telephone"])
        day_of_week = st.number_input("Last Contact Day of Month", value=15)
        month = st.selectbox(
            "Last Contact Month",
            ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        )
        duration = st.number_input("Last Contact Duration in Seconds", value=300)
        campaign = st.number_input("Number of Contacts During Campaign", value=2)
        pdays = st.number_input("Days Since Previous Contact", value=-1)
        previous = st.number_input("Previous Contacts", value=0)
        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "other", "success"])

    if st.button("Predict Subscription", type="primary"):
        input_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day_of_week": day_of_week,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
        }
        result = predict_subscription_cloud_or_local(input_data)
        st.caption(f"Inference source: **{result.get('inference_source', 'unknown')}**")
        if result.get("warning"):
            st.warning(result["warning"])
        st.metric("Subscription Prediction", result["prediction"].upper())
        st.metric("Probability of Yes", f"{result['probability_yes'] * 100:.2f}%")
        st.metric("Probability of No", f"{result['probability_no'] * 100:.2f}%")
        st.json(result)
