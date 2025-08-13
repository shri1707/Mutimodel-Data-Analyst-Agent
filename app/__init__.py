"""
Data Analysis API Package

A FastAPI-based system for file-driven data analysis with AI code generation.
"""

__version__ = "1.0.0"
__author__ = "Data Analysis API Team"
__description__ = "AI-powered file analysis system"

# Package exports for Phase 2
from .main import app
from .logger import setup_logger, logger
from .utils import infer_task_type, preview_file, sanitize_filename

__all__ = [
    "app",
    "setup_logger",
    "logger",
    "infer_task_type",
    "preview_file", 
    "sanitize_filename",
]