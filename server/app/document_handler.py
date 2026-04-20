"""
Document Handler Module
Handles file uploads and text extraction from PDF, TXT, DOCX files.
Extracted text is used as reference context for blog generation agents.
"""

from __future__ import annotations

import os
import io
import tempfile
import shutil
from typing import List, Optional
from pathlib import Path

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file
MAX_FILES = 5


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text content from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []

    for page_num, page in enumerate(reader.pages, 1):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            text_parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")

    return "\n\n".join(text_parts)


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text content from a DOCX file."""
    from docx import Document

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = []

    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text.strip())

    return "\n\n".join(paragraphs)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text content from a plain text or markdown file."""
    import chardet

    detected = chardet.detect(file_bytes)
    encoding = detected.get("encoding", "utf-8") or "utf-8"

    try:
        return file_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return file_bytes.decode("utf-8", errors="ignore")


def extract_text(filename: str, file_bytes: bytes) -> str:
    """
    Extract text from a file based on its extension.
    
    Args:
        filename: Original filename with extension
        file_bytes: Raw file bytes
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file type is not supported
    """
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".docx":
        return extract_text_from_docx(file_bytes)
    elif ext in (".txt", ".md"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError(f"No handler for extension: {ext}")


async def process_uploaded_files(files) -> str:
    """
    Process multiple uploaded files and combine their text content.
    
    Args:
        files: List of UploadFile objects from FastAPI
        
    Returns:
        Combined text content from all files, formatted with headers
    """
    if not files:
        return ""

    if len(files) > MAX_FILES:
        raise ValueError(f"Maximum {MAX_FILES} files allowed per request")

    combined_texts = []

    for file in files:
        # Read file content
        file_bytes = await file.read()

        # Check file size
        if len(file_bytes) > MAX_FILE_SIZE:
            raise ValueError(
                f"File '{file.filename}' exceeds maximum size of "
                f"{MAX_FILE_SIZE // (1024 * 1024)}MB"
            )

        # Skip empty files
        if len(file_bytes) == 0:
            continue

        # Extract text
        try:
            text = extract_text(file.filename, file_bytes)
            if text.strip():
                combined_texts.append(
                    f"=== Document: {file.filename} ===\n{text.strip()}"
                )
        except Exception as e:
            combined_texts.append(
                f"=== Document: {file.filename} (Error: {str(e)}) ==="
            )

    return "\n\n" + "\n\n".join(combined_texts) if combined_texts else ""


def format_context_for_agents(document_text: str) -> str:
    """
    Format extracted document text as context for the agents.
    
    Args:
        document_text: Combined text from uploaded documents
        
    Returns:
        Formatted context string for prompt injection
    """
    if not document_text or not document_text.strip():
        return ""

    return (
        "\n\n---\n"
        "## Reference Documents (uploaded by user)\n"
        "Use the following documents as primary reference material for the blog post. "
        "Incorporate relevant information, data, and insights from these documents.\n\n"
        f"{document_text}\n"
        "---\n"
    )
