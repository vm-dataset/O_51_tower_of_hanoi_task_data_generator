"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK CONFIGURATION                             ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define your task-specific settings.                   ║
║  Inherits common settings from core.GenerationConfig                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import Optional
from pydantic import Field
from core import GenerationConfig


class TaskConfig(GenerationConfig):
    """
    Your task-specific configuration.
    
    CUSTOMIZE THIS CLASS to add your task's hyperparameters.
    
    Inherited from GenerationConfig:
        - num_samples: int          # Number of samples to generate
        - domain: str               # Task domain name
        - difficulty: Optional[str] # Difficulty level
        - random_seed: Optional[int] # For reproducibility
        - output_dir: Path          # Where to save outputs
        - image_size: tuple[int, int] # Image dimensions
    """
    
    # ══════════════════════════════════════════════════════════════════════════
    #  OVERRIDE DEFAULTS
    # ══════════════════════════════════════════════════════════════════════════
    
    domain: str = Field(default="tower_of_hanoi")
    image_size: tuple[int, int] = Field(default=(512, 512))
    
    # ══════════════════════════════════════════════════════════════════════════
    #  VIDEO SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    generate_videos: bool = Field(
        default=True,
        description="Whether to generate ground truth videos"
    )
    
    video_fps: int = Field(
        default=10,
        description="Video frame rate"
    )

    hold_frames: int = Field(
        default=2,
        description="Frames to hold at the start/end of the video and between moves (lower = faster/smaller videos)"
    )

    transition_frames: int = Field(
        default=8,
        description="Frames for each disk movement transition (lower = faster/smaller videos)"
    )

    ensure_unique: bool = Field(
        default=True,
        description="Ensure each generated task is unique within a single generation run (by initial state + num_disks)"
    )

    max_unique_attempts_per_task: int = Field(
        default=500,
        description="Max resampling attempts per task to satisfy uniqueness"
    )
    
    # ══════════════════════════════════════════════════════════════════════════
    #  TASK-SPECIFIC SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    
    # Add your custom settings here
    num_disks: Optional[int] = Field(
        default=None,
        description="Fixed number of disks in Tower of Hanoi (3-5). If None, uses random between min_disks and max_disks."
    )
    
    min_disks: int = Field(
        default=3,
        description="Minimum number of disks for random generation"
    )
    
    max_disks: int = Field(
        default=5,
        description="Maximum number of disks for random generation"
    )
    
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.4, "medium": 0.4, "hard": 0.2},
        description="Distribution of difficulty levels"
    )
