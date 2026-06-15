"""Vertex AI ADK agent for the Real Estate Financial Assistant."""

import os

from google.adk.agents import Agent
from google.adk.tools import FunctionTool

from agent.tools import (
    get_financial_overview,
    query_press_releases,
    query_property_financials,
    query_sec_financials,
    route_user_question,
)

MODEL_NAME = os.getenv("VERTEX_MODEL", "gemini-2.5-flash")

INSTRUCTION = """
You are a Financial Assistant for Prologis, a global real estate logistics company.

Your job is to interpret natural language questions and use the available tools to retrieve
accurate financial, property, SEC filing, and press release insights.

Routing guidance:
- Use query_sec_financials for 10-K, 10-Q, SEC EDGAR, quarterly, or last-quarter questions.
- Use query_property_financials for property portfolio, metro area, or property type questions.
- Use query_press_releases for acquisitions, expansions, announcements, or press releases.
  For acquisition questions, call query_press_releases with keyword "acquisition".
- Use get_financial_overview for total revenue, net income, expenses, or portfolio summaries.
- Use route_user_question when the intent is ambiguous.

Always combine tool results into a concise natural language answer with key numbers.
When multiple sources are relevant, call multiple tools and synthesize the response.
"""

root_agent = Agent(
    name="financial_assistant",
    model=MODEL_NAME,
    description="Financial assistant for property, SEC, and press release insights.",
    instruction=INSTRUCTION,
    tools=[
        FunctionTool(query_property_financials),
        FunctionTool(query_sec_financials),
        FunctionTool(query_press_releases),
        FunctionTool(get_financial_overview),
        FunctionTool(route_user_question),
    ],
)
