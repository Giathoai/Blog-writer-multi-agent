"""
Crew Orchestration Module
Sequential pipeline: Planner → Writer → Editor
Each agent's output feeds into the next agent as input.
"""

from __future__ import annotations

import logging
from typing import Optional

from app.agents import (
    build_planner_chain,
    build_writer_chain,
    build_editor_chain,
    search_web,
)
from app.document_handler import format_context_for_agents

logger = logging.getLogger(__name__)


class BlogWriterCrew:
    """
    Multi-agent blog writing crew.
    
    Pipeline:
        1. Planner: Creates content outline (with web search + uploaded docs)
        2. Writer: Writes full blog draft from outline
        3. Editor: Polishes and refines the draft
    """

    def __init__(self):
        logger.info("Initializing Blog Writer Crew...")
        self.planner_chain = build_planner_chain()
        self.writer_chain = build_writer_chain()
        self.editor_chain = build_editor_chain()
        logger.info("Blog Writer Crew initialized successfully")

    def run(self, topic: str, document_text: str = "") -> dict:
        """
        Execute the full blog writing pipeline.
        
        Args:
            topic: The blog topic to write about
            document_text: Optional text from uploaded documents
            
        Returns:
            Dict with keys: topic, plan, draft, final, steps
        """
        steps = []

        # Format document context
        doc_context = format_context_for_agents(document_text)

        # --- Step 1: Web Search (if available) ---
        logger.info(f"[Step 0] Searching web for: {topic}")
        search_results = search_web(topic)
        steps.append({"agent": "Search", "status": "complete"})

        # Combine search results with document context
        combined_context = ""
        if search_results and "unavailable" not in search_results.lower():
            combined_context += f"\n\nWeb Search Results:\n{search_results}\n"
        if doc_context:
            combined_context += doc_context

        # --- Step 2: Planner Agent ---
        logger.info(f"[Step 1] Planner agent working on: {topic}")
        steps.append({"agent": "Planner", "status": "running"})

        try:
            content_plan = self.planner_chain.invoke({
                "topic": topic,
                "document_context": combined_context,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 1] Planner agent completed")
        except Exception as e:
            logger.error(f"[Step 1] Planner failed: {e}")
            steps[-1]["status"] = "failed"
            raise RuntimeError(f"Planner agent failed: {str(e)}")

        # --- Step 3: Writer Agent ---
        logger.info(f"[Step 2] Writer agent working on: {topic}")
        steps.append({"agent": "Writer", "status": "running"})

        try:
            blog_draft = self.writer_chain.invoke({
                "topic": topic,
                "content_plan": content_plan,
                "document_context": doc_context,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 2] Writer agent completed")
        except Exception as e:
            logger.error(f"[Step 2] Writer failed: {e}")
            steps[-1]["status"] = "failed"
            raise RuntimeError(f"Writer agent failed: {str(e)}")

        # --- Step 4: Editor Agent ---
        logger.info(f"[Step 3] Editor agent working on: {topic}")
        steps.append({"agent": "Editor", "status": "running"})

        try:
            final_blog = self.editor_chain.invoke({
                "topic": topic,
                "blog_draft": blog_draft,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 3] Editor agent completed")
        except Exception as e:
            logger.error(f"[Step 3] Editor failed: {e}")
            steps[-1]["status"] = "failed"
            # If editor fails, return the draft as fallback
            final_blog = blog_draft
            steps[-1]["status"] = "failed_with_fallback"

        # Clean up markdown output
        final_blog = _clean_markdown(final_blog)

        return {
            "topic": topic,
            "plan": content_plan,
            "draft": blog_draft,
            "final": final_blog,
            "steps": steps,
        }


def _clean_markdown(text: str) -> str:
    """Clean up common LLM output artifacts from markdown."""
    if not text:
        return text

    text = text.strip()

    # Remove leading ```markdown or ``` blocks
    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    elif text.startswith("```md"):
        text = text[len("```md"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    # Remove trailing ``` 
    if text.endswith("```"):
        text = text[:-3].strip()

    # Remove leading "markdown" word
    if text.lower().startswith("markdown\n"):
        text = text[9:].strip()

    return text
