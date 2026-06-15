"""Run Vertex AI ADK agent with graceful local fallback."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

from app.chat_router import route_question


def _vertex_adk_enabled() -> bool:
    if os.getenv("USE_VERTEX_ADK", "false").lower() != "true":
        return False
    return bool(os.getenv("GOOGLE_CLOUD_PROJECT"))


def _extract_text_from_events(events) -> str:
    parts: list[str] = []
    for event in events:
        content = getattr(event, "content", None)
        if not content or not getattr(content, "parts", None):
            continue
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                parts.append(text.strip())
    return "\n".join(part for part in parts if part).strip()


def ask_vertex_adk(question: str, session_id: str | None = None) -> dict[str, Any]:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    from agent.financial_assistant_agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="real-estate-financial-assistant",
        session_service=session_service,
    )

    user_id = "streamlit-user"
    session_id = session_id or str(uuid.uuid4())

    try:
        from google.genai import types
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "google-genai is required for Vertex AI ADK. "
            "Install it with `pip install google-genai` or disable USE_VERTEX_ADK."
        ) from exc

    session_service.create_session_sync(
        app_name="real-estate-financial-assistant",
        user_id=user_id,
        session_id=session_id,
    )

    events = runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=question)],
        ),
    )

    answer = _extract_text_from_events(events)
    routed = route_question(question)
    routed["answer"] = answer or routed["answer"]
    routed["engine"] = "vertex_ai_adk"
    routed["session_id"] = session_id
    return routed


def ask_assistant(question: str, session_id: str | None = None) -> dict[str, Any]:
    if _vertex_adk_enabled():
        try:
            return ask_vertex_adk(question, session_id=session_id)
        except Exception as exc:
            fallback = route_question(question)
            fallback["engine"] = "local_fallback"
            fallback["warning"] = f"Vertex AI ADK unavailable: {exc}"
            return fallback

    response = route_question(question)
    response["engine"] = "local_router"
    return response


def response_to_json(response: dict[str, Any]) -> str:
    payload = dict(response)
    dataframe = payload.pop("dataframe", None)
    if dataframe is not None:
        payload["records"] = dataframe.to_dict(orient="records")
    return json.dumps(payload, indent=2, default=str)
