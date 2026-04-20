"""
Agent Definitions and Web Search Tool
Defines the 3 agents (Planner, Writer, Editor) and the search tool.
"""

from __future__ import annotations

import os
from typing import Optional

from langchain_core.output_parsers import StrOutputParser

from app.llm_config import get_llm
from app.tasks import PLANNER_PROMPT, WRITER_PROMPT, EDITOR_PROMPT


def get_search_tool():
    """
    Create Google Serper search tool for web research.
    Returns None if SERPER_API_KEY is not configured.
    """
    serper_key = os.getenv("SERPER_API_KEY", "")

    if not serper_key or serper_key == "your_serper_api_key_here":
        return None

    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper

        search = GoogleSerperAPIWrapper(
            serper_api_key=serper_key,
            k=5,  # Number of results
        )
        return search
    except Exception:
        return None


def search_web(query: str) -> str:
    """
    Perform a web search and return results as formatted text.
    Falls back gracefully if search is unavailable.
    """
    search = get_search_tool()

    if search is None:
        return "(Web search unavailable - no SERPER_API_KEY configured)"

    try:
        results = search.run(query)
        return f"Web search results for '{query}':\n{results}"
    except Exception as e:
        return f"(Web search failed: {str(e)})"


def build_planner_chain():
    """Build the Content Planner agent chain."""
    llm = get_llm()
    return PLANNER_PROMPT | llm | StrOutputParser()


def build_writer_chain():
    """Build the Content Writer agent chain."""
    llm = get_llm()
    return WRITER_PROMPT | llm | StrOutputParser()


def build_editor_chain():
    """Build the Editor agent chain."""
    llm = get_llm()
    return EDITOR_PROMPT | llm | StrOutputParser()
