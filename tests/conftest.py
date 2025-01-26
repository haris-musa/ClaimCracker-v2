"""Pytest configuration file."""

import pytest
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def pytest_configure(config):
    """Configure pytest."""
    # Add project markers
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running"
    )

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    monkeypatch.setenv("TF_ENABLE_ONEDNN_OPTS", "0")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "2")
    monkeypatch.setenv("TESTING", "1")  # Indicate test environment 