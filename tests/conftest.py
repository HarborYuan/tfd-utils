"""
Pytest configuration for tfd_utils tests.
"""

import pytest
import os
import sys

# Add src directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide a temporary directory for test data."""
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp(prefix="tfd_utils_pytest_")
    yield test_dir
    shutil.rmtree(test_dir)
