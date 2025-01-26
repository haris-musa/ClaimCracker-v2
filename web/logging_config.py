"""Logging configuration for the API."""

import logging
import logging.handlers
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class StructuredJsonFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format with extra fields."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        # Get the standard fields
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name
        }
        
        # Add extra fields if they exist
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_data.update(record.extra_data)
            
        # Add exception info if it exists
        if record.exc_info:
            log_data["exc_info"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logging() -> logging.Logger:
    """Set up structured logging with console and file handlers."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("claimcracker")
    logger.setLevel(logging.INFO)
    
    # Create formatters
    json_formatter = StructuredJsonFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        filename=logs_dir / "api.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # File handler for errors
    error_handler = logging.handlers.RotatingFileHandler(
        filename=logs_dir / "errors.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    error_handler.setFormatter(json_formatter)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    return logger

# Create and configure logger
logger = setup_logging() 