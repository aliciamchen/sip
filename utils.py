#!/usr/bin/env python3
"""
Utility functions for the saliva-inverse-planning project.
"""

from pathlib import Path


def get_project_root():
    """Find the project root directory by looking for the .git directory."""
    current_dir = Path(__file__).resolve()
    
    # Look for .git directory (most reliable indicator)
    for parent in current_dir.parents:
        if (parent / '.git').exists():
            return parent
    
    # Fallback: assume current directory is project root
    return current_dir.parent
