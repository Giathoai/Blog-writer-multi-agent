from __future__ import annotations

import logging
from typing import Optional

from app.agents import (
    build_analyzer_chain,  # Import thêm chain mới
    build_planner_chain,
    build_writer_chain,
    build_editor_chain,
    search_web,
    build_reviser_chain,
)
from app.document_handler import format_context_for_agents

logger = logging.getLogger(__name__)

class BlogWriterCrew:
    def __init__(self):
        logger.info("Initializing Blog Writer Crew...")
        self.analyzer_chain = build_analyzer_chain() # Khởi tạo Analyzer
        self.planner_chain = build_planner_chain()
        self.writer_chain = build_writer_chain()
        self.editor_chain = build_editor_chain()
        logger.info("Blog Writer Crew initialized successfully")

    def run(self, topic: str, document_text: str = "") -> dict:
        steps = []
        doc_context = format_context_for_agents(document_text)

        # --- Bước 0: Analyzer Agent phân tích yêu cầu ---
        logger.info(f"[Step 0] Analyzing request: {topic}")
        steps.append({"agent": "Analyzer", "status": "running"})
        try:
            analysis = self.analyzer_chain.invoke({
                "topic": topic,
                "document_context": doc_context if doc_context else "Không có tài liệu đính kèm."
            })
            refined_topic = analysis.get("refined_topic", topic)
            needs_search = analysis.get("needs_search", False)
            search_query = analysis.get("search_query", "")
            steps[-1]["status"] = "complete"
            logger.info(f"[Step 0] Analyzer done. Refined Topic: '{refined_topic}', Needs Search: {needs_search}")
        except Exception as e:
            logger.error(f"[Step 0] Analyzer failed: {e}")
            # Fallback nếu LLM trả JSON lỗi
            refined_topic = topic
            needs_search = True if not doc_context else False
            search_query = topic
            steps[-1]["status"] = "failed_with_fallback"

        combined_context = ""

        if needs_search and search_query:
            logger.info(f"[Step 1] Searching web for: {search_query}")
            search_results = search_web(search_query)
            steps.append({"agent": "Search", "status": "complete"})
            if search_results and "unavailable" not in search_results.lower():
                combined_context += f"\n\n--- KẾT QUẢ TÌM KIẾM WEB ---\n{search_results}\n"
        else:
            logger.info("[Step 1] Skipping web search based on document context.")
            steps.append({"agent": "Search", "status": "skipped"})

        if doc_context:
            combined_context += f"\n\n--- TÀI LIỆU CỦA NGƯỜI DÙNG ---\n{doc_context}\n"

        logger.info(f"[Step 2] Planner agent working on: {refined_topic}")
        steps.append({"agent": "Planner", "status": "running"})

        try:
            content_plan = self.planner_chain.invoke({
                "topic": refined_topic,
                "document_context": combined_context,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 2] Planner agent completed")
        except Exception as e:
            logger.error(f"[Step 2] Planner failed: {e}")
            steps[-1]["status"] = "failed"
            raise RuntimeError(f"Planner agent failed: {str(e)}")

        logger.info(f"[Step 3] Writer agent working on: {refined_topic}")
        steps.append({"agent": "Writer", "status": "running"})

        try:
            blog_draft = self.writer_chain.invoke({
                "topic": refined_topic,
                "content_plan": content_plan,
                "document_context": combined_context,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 3] Writer agent completed")
        except Exception as e:
            logger.error(f"[Step 3] Writer failed: {e}")
            steps[-1]["status"] = "failed"
            raise RuntimeError(f"Writer agent failed: {str(e)}")

        logger.info(f"[Step 4] Editor agent working on: {refined_topic}")
        steps.append({"agent": "Editor", "status": "running"})

        try:
            final_blog = self.editor_chain.invoke({
                "topic": refined_topic,
                "blog_draft": blog_draft,
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 4] Editor agent completed")
        except Exception as e:
            logger.error(f"[Step 4] Editor failed: {e}")
            steps[-1]["status"] = "failed"
            final_blog = blog_draft
            steps[-1]["status"] = "failed_with_fallback"

        final_blog = _clean_markdown(final_blog)

        return {
            "topic": refined_topic, 
            "plan": content_plan,
            "draft": blog_draft,
            "final": final_blog,
            "steps": steps,
        }
    def revise(self, current_blog: str, comments_data: list[dict]) -> str:
        """
        Takes the current blog and a list of specific comments, returns revised blog.
        """
        logger.info("Reviser agent is processing user comments...")
        
        formatted_comments = ""
        for idx, item in enumerate(comments_data):
            formatted_comments += f"\nComment #{idx + 1}:\n"
            formatted_comments += f"- Đoạn văn bản được chọn: \"{item['selected_text']}\"\n"
            formatted_comments += f"- Yêu cầu sửa: {item['comment']}\n"

        try:
            revised_blog = self.reviser_chain.invoke({
                "current_blog": current_blog,
                "user_comments": formatted_comments
            })
            logger.info("Reviser agent completed successfully.")
            return _clean_markdown(revised_blog)
        except Exception as e:
            logger.error(f"Reviser failed: {e}")
            raise RuntimeError(f"Reviser agent failed: {str(e)}")

def _clean_markdown(text: str) -> str:
    """Clean up common LLM output artifacts from markdown."""
    if not text:
        return text

    text = text.strip()

    if text.startswith("```markdown"):
        text = text[len("```markdown"):].strip()
    elif text.startswith("```md"):
        text = text[len("```md"):].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    if text.lower().startswith("markdown\n"):
        text = text[9:].strip()

    return text
