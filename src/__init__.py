"""
Tower of Hanoi task implementation.

Task components:
    - config.py   : Tower of Hanoi task configuration (TaskConfig)
    - generator.py: Tower of Hanoi generation logic (TaskGenerator)
    - prompts.py  : Tower of Hanoi task prompts/instructions (get_prompt)
"""

from .config import TaskConfig
from .generator import TaskGenerator
from .prompts import get_prompt

__all__ = ["TaskConfig", "TaskGenerator", "get_prompt"]
