"""ADK tool functions for the Financial Assistant agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "app"))

from chat_router import route_question  # noqa: E402
from db import filter_properties, get_portfolio_summary, get_property_financials  # noqa: E402
from press_release_loader import search_press_releases  # noqa: E402
from sec_financials import get_latest_sec_summary, load_sec_financials  # noqa: E402


def query_property_financials(
    metro_area: str = "All",
    property_type: str = "All",
) -> str:
    """Query property and financial records from the Postgres/SQLite database.

    metro_area must be a city name such as Chicago, Dallas, or Los Angeles
    (do not include words like 'region' or 'area').
    property_type must be Industrial, Warehouse, or Office.
    """
    result = filter_properties(metro_area, property_type)
    summary = get_portfolio_summary()
    payload = {
        "filters": {"metro_area": metro_area, "property_type": property_type},
        "count": len(result),
        "total_revenue": float(result["revenue"].sum()) if not result.empty else 0.0,
        "total_net_income": float(result["net_income"].sum()) if not result.empty else 0.0,
        "portfolio_summary": summary,
        "records": result.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def query_sec_financials(period: str = "latest") -> str:
    """Retrieve SEC EDGAR 10-K and 10-Q financial metrics such as revenue and net income."""
    df = load_sec_financials()
    if period != "latest":
        filtered = df[df["period"].str.lower() == period.lower()]
        if not filtered.empty:
            df = filtered
    payload = {
        "summary": get_latest_sec_summary(),
        "filings": df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def query_press_releases(keyword: str = "") -> str:
    """Search company press releases for acquisitions, expansions, and business updates.

    For acquisition questions, pass keyword='acquisition' or 'acquisitions'.
    For expansion questions, pass keyword='expansion'.
    """
    df = search_press_releases(keyword)
    payload = {
        "keyword": keyword,
        "count": len(df),
        "releases": df.to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def get_financial_overview() -> str:
    """Return portfolio-level revenue, net income, expense, and property count metrics."""
    summary = get_portfolio_summary()
    df = get_property_financials()
    payload = {
        "summary": summary,
        "top_properties_by_revenue": df.nlargest(5, "revenue").to_dict(orient="records"),
    }
    return json.dumps(payload, indent=2)


def route_user_question(question: str) -> str:
    """Route a natural language question to the correct structured data source."""
    response = route_question(question)
    if response.get("dataframe") is not None:
        response["records"] = response["dataframe"].to_dict(orient="records")
        del response["dataframe"]
    return json.dumps(response, indent=2)
