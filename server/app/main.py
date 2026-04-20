"""
FastAPI Main Application
Blog Writer Multi-Agent API Server

Endpoints:
    POST /generate-blog/     - Generate blog with optional file uploads
    GET  /health              - Health check  
    GET  /config              - Current LLM configuration
"""

from __future__ import annotations
import uuid 
import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.crew import BlogWriterCrew
from app.document_handler import process_uploaded_files
from app.llm_config import get_llm_info
SESSIONS_STORE = {}
# Load environment
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

class UserComment(BaseModel):
    selected_text: str
    comment: str

class ReviseRequest(BaseModel):
    comments: List[UserComment]
# --- Lifespan (startup/shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize crew on startup."""
    logger.info("Starting Blog Writer Multi-Agent Server...")
    try:
        app.state.crew = BlogWriterCrew()
        llm_info = get_llm_info()
        logger.info(f"LLM Provider: {llm_info['provider']} | Model: {llm_info['model']}")
        logger.info("Server ready!")
    except Exception as e:
        logger.error(f"Failed to initialize crew: {e}")
        app.state.crew = None

    yield

    logger.info("Shutting down server...")


# --- FastAPI App ---
app = FastAPI(
    title="Blog Writer Multi-Agent API",
    description="Multi-agent blog writing system powered by LangChain and AI",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in cors_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---
class TopicRequest(BaseModel):
    topic: str


class BlogResponse(BaseModel):
    topic: str
    blog: Dict[str, Any]


# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    crew_ready = app.state.crew is not None
    llm_info = get_llm_info()
    return {
        "status": "healthy" if crew_ready else "degraded",
        "crew_ready": crew_ready,
        "llm": llm_info,
    }


@app.get("/config")
async def get_config():
    """Return current LLM configuration."""
    return get_llm_info()


@app.post("/generate-blog/")
async def generate_blog(
    topic: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
):
    """
    Generate a blog post using the multi-agent pipeline.
    
    - **topic**: The blog topic (required)
    - **files**: Optional reference documents (PDF, TXT, DOCX, MD - max 5 files)
    
    The pipeline:
    1. Web search for latest info on the topic
    2. Planner agent creates content outline
    3. Writer agent drafts the blog post
    4. Editor agent polishes the final version
    """
    if not topic or not topic.strip():
        raise HTTPException(status_code=400, detail="'topic' must be provided")

    if app.state.crew is None:
        raise HTTPException(
            status_code=503,
            detail="Blog writer crew not initialized. Check server logs."
        )

    # Process uploaded documents
    document_text = ""
    if files:
        try:
            document_text = await process_uploaded_files(files)
            if document_text:
                logger.info(f"Processed {len(files)} uploaded file(s)")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.warning(f"Document processing warning: {e}")

    # Generate blog
    try:
        logger.info(f"Generating blog for topic: '{topic.strip()}'")
        result = app.state.crew.run(
            topic=topic.strip(),
            document_text=document_text,
        )

        # Tạo Session ID và lưu lại trạng thái bài blog
        session_id = str(uuid.uuid4())
        SESSIONS_STORE[session_id] = {
            "topic": result["topic"],
            "blog_content": result["final"],
            "history": [{"role": "system", "action": "created"}]
        }

        return {
            "session_id": session_id, # Trả về Session ID để UI dùng cho các api sau
            "topic": result["topic"],
            "blog": {
                "raw": result["final"],
                "plan": result["plan"],
                "draft": result["draft"],
            },
            "steps": result["steps"],
        }
    

    except RuntimeError as e:
        logger.error(f"Blog generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/generate-blog-json/")
async def generate_blog_json(request: TopicRequest):
    """
    Simple JSON endpoint for blog generation (no file upload).
    Compatible with the original API format.
    
    Body: { "topic": "Your Topic" }
    """
    if not request.topic or not request.topic.strip():
        raise HTTPException(status_code=400, detail="'topic' must be provided")

    if app.state.crew is None:
        raise HTTPException(
            status_code=503,
            detail="Blog writer crew not initialized. Check server logs."
        )

    try:
        logger.info(f"Generating blog (JSON) for topic: '{request.topic.strip()}'")
        result = app.state.crew.run(topic=request.topic.strip())

        return {
            "topic": result["topic"],
            "blog": {"raw": result["final"]},
        }

    except Exception as e:
        logger.error(f"Blog generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/revise-blog/{session_id}")
async def revise_blog(session_id: str, request: ReviseRequest):
    """
    Endpoint nhận feedback từ người dùng trên các đoạn văn bản cụ thể.
    """
    if app.state.crew is None:
        raise HTTPException(status_code=503, detail="Crew not initialized.")

    if session_id not in SESSIONS_STORE:
        raise HTTPException(status_code=404, detail="Session không tồn tại hoặc đã hết hạn.")

    session_data = SESSIONS_STORE[session_id]
    current_blog = session_data["blog_content"]

    try:
        logger.info(f"Revising blog for session: {session_id} with {len(request.comments)} comments.")
        
        # Gọi hàm revise trong crew
        comments_dict_list = [c.dict() for c in request.comments]
        revised_blog = app.state.crew.revise(current_blog, comments_dict_list)

        # Cập nhật lại kho lưu trữ với bản mới nhất
        SESSIONS_STORE[session_id]["blog_content"] = revised_blog
        SESSIONS_STORE[session_id]["history"].append({
            "role": "user",
            "comments": comments_dict_list
        })
        SESSIONS_STORE[session_id]["history"].append({
            "role": "system",
            "action": "revised"
        })

        return {
            "session_id": session_id,
            "blog": {
                "raw": revised_blog
            },
            "message": "Đã cập nhật bài blog dựa trên nhận xét của bạn."
        }

    except Exception as e:
        logger.error(f"Blog revision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))