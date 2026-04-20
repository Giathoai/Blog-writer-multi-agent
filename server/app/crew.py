from __future__ import annotations

import logging
from typing import Optional

from app.agents import (
    build_analyzer_chain,  # Import thêm chain mới
    build_planner_chain,
    build_writer_chain,
    build_editor_chain,
    search_web,
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

        # Cấu trúc lại Context tổng hợp
        combined_context = ""

        # --- Bước 1: Web Search (Chỉ chạy khi Analyzer bảo cần) ---
        if needs_search and search_query:
            logger.info(f"[Step 1] Searching web for: {search_query}")
            search_results = search_web(search_query)
            steps.append({"agent": "Search", "status": "complete"})
            if search_results and "unavailable" not in search_results.lower():
                combined_context += f"\n\n--- KẾT QUẢ TÌM KIẾM WEB ---\n{search_results}\n"
        else:
            logger.info("[Step 1] Skipping web search based on document context.")
            steps.append({"agent": "Search", "status": "skipped"})

        # Gắn thêm document context vào 
        if doc_context:
            combined_context += f"\n\n--- TÀI LIỆU CỦA NGƯỜI DÙNG ---\n{doc_context}\n"

        # --- Bước 2: Planner Agent ---
        # Lưu ý: Truyền 'refined_topic' thay vì 'topic' rác ban đầu
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

        # --- Bước 3: Writer Agent ---
        logger.info(f"[Step 3] Writer agent working on: {refined_topic}")
        steps.append({"agent": "Writer", "status": "running"})

        try:
            blog_draft = self.writer_chain.invoke({
                "topic": refined_topic,
                "content_plan": content_plan,
                "document_context": combined_context, # Đưa đầy đủ context cho writer
            })
            steps[-1]["status"] = "complete"
            logger.info("[Step 3] Writer agent completed")
        except Exception as e:
            logger.error(f"[Step 3] Writer failed: {e}")
            steps[-1]["status"] = "failed"
            raise RuntimeError(f"Writer agent failed: {str(e)}")

        # --- Bước 4: Editor Agent ---
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
            "topic": refined_topic, # Trả về chủ đề đã được làm gọn
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
