"""
Configuration for pytest integration tests.
This file sets up common fixtures and configuration for integration tests.
"""

import sys
import os
import pytest

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# Also add the agent_suite directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_suite"))) 