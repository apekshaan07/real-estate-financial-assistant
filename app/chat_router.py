"""Rule-based query router used as local fallback and for structured results."""

from __future__ import annotations

import re

import pandas as pd

from db import filter_properties, get_portfolio_summary, get_property_financials
from press_release_loader import load_press_releases, search_press_releases
from sec_financials import get_latest_sec_summary, load_sec_financials


def _extract_metro(question: str) -> str | None:
    q = question.lower()
    metros = [
        "Chicago", "Dallas", "Los Angeles", "Atlanta", "Phoenix", "New York",
        "San Francisco", "Austin", "Seattle", "Denver", "Miami", "Houston",
        "Columbus", "Boston", "Indianapolis", "Memphis", "Charlotte", "Nashville",
    ]
    for metro in metros:
        if metro.lower() in q:
            return metro
    return None


def _extract_property_type(question: str) -> str | None:
    q = question.lower()
    types = ["Industrial", "Warehouse", "Office"]
    for property_type in types:
        if property_type.lower() in q:
            return property_type
    return None


def route_question(question: str) -> dict:
    """Return structured chat response with answer text and optional data payload."""
    q = question.lower().strip()
    portfolio = get_portfolio_summary()

    if "chicago" in q and "industrial" in q:
        result = filter_properties("Chicago", "Industrial")
        return {
            "source": "properties",
            "answer": (
                f"I found {len(result)} industrial properties in Chicago with total revenue of "
                f"${result['revenue'].sum():,.0f} and net income of ${result['net_income'].sum():,.0f}."
            ),
            "dataframe": result,
            "data_type": "Property Financials",
        }

    metro = _extract_metro(question)
    property_type = _extract_property_type(question)
    if metro or property_type:
        result = filter_properties(metro, property_type)
        filters = []
        if metro:
            filters.append(metro)
        if property_type:
            filters.append(property_type)
        label = " / ".join(filters)
        return {
            "source": "properties",
            "answer": (
                f"I found {len(result)} properties matching {label}. "
                f"Total revenue is ${result['revenue'].sum():,.0f}."
            ),
            "dataframe": result,
            "data_type": "Property Financials",
        }

    if "acquisition" in q or "acquisitions" in q:
        press_df = search_press_releases("acquisition")
        answer = "No recent acquisition press releases were found."
        if not press_df.empty:
            latest = press_df.iloc[0]
            answer = (
                f"Yes. The company announced acquisition activity in '{latest['title']}'. "
                f"{latest['summary']}"
            )
        return {
            "source": "press_releases",
            "answer": answer,
            "dataframe": press_df,
            "data_type": "Press Releases",
        }

    if "expansion" in q or "expand" in q:
        press_df = search_press_releases("expansion")
        answer = "No expansion press releases were found."
        if not press_df.empty:
            latest = press_df.iloc[0]
            answer = (
                f"The company reported expansion activity in '{latest['title']}'. "
                f"{latest['summary']}"
            )
        return {
            "source": "press_releases",
            "answer": answer,
            "dataframe": press_df,
            "data_type": "Press Releases",
        }

    if any(token in q for token in ["sec", "10-k", "10-q", "filing", "last quarter", "quarterly filing"]):
        sec_df = load_sec_financials()
        return {
            "source": "sec_edgar",
            "answer": get_latest_sec_summary(),
            "dataframe": sec_df,
            "data_type": "SEC EDGAR Financials",
        }

    if "press release" in q or "announcement" in q or "announced" in q:
        press_df = load_press_releases()
        return {
            "source": "press_releases",
            "answer": (
                f"There are {len(press_df)} recent press releases covering acquisitions, "
                "expansions, quarterly updates, and sustainability initiatives."
            ),
            "dataframe": press_df,
            "data_type": "Press Releases",
        }

    if re.search(r"\bnet income\b", q):
        return {
            "source": "portfolio",
            "answer": f"Total portfolio net income is ${portfolio['total_net_income']:,.0f}.",
            "dataframe": get_property_financials()[["property_id", "address", "net_income"]],
            "data_type": "Net Income Summary",
        }

    if re.search(r"\brevenue\b", q):
        return {
            "source": "portfolio",
            "answer": f"Total portfolio revenue is ${portfolio['total_revenue']:,.0f}.",
            "dataframe": get_property_financials()[["property_id", "address", "revenue"]],
            "data_type": "Revenue Summary",
        }

    if "expense" in q or "expenses" in q:
        return {
            "source": "portfolio",
            "answer": f"Total portfolio expenses are ${portfolio['total_expenses']:,.0f}.",
            "dataframe": get_property_financials()[["property_id", "address", "expenses"]],
            "data_type": "Expense Summary",
        }

    if "property" in q or "properties" in q:
        df = get_property_financials()
        return {
            "source": "properties",
            "answer": f"The portfolio contains {len(df)} properties across multiple metro areas.",
            "dataframe": df,
            "data_type": "Property Financials",
        }

    return {
        "source": "general",
        "answer": (
            "I can help with property financials, SEC filings, press releases, acquisitions, "
            "expansions, revenue, net income, and expenses. Try asking about Chicago industrial "
            "properties or last quarter net income."
        ),
        "dataframe": None,
        "data_type": "Help",
    }
