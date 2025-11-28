"""
Utility helpers for RAG backend.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from .config import DATA_DIR


def save_upload_file_tmp(upload_file: UploadFile) -> Path:
    """
    Persist uploaded file to a temporary location on disk.
    """
    suffix = Path(upload_file.filename or "upload.pdf").suffix or ".pdf"
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    upload_dir = DATA_DIR / "tmp_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / temp_name
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return temp_path


def cleanup_file(file_path: Optional[Path]) -> None:
    """Delete temporary file if it exists."""
    if not file_path:
        return
    try:
        Path(file_path).unlink(missing_ok=True)
    except OSError:
        pass


__all__ = ["save_upload_file_tmp", "cleanup_file"]


