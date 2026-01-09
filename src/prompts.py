"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK PROMPTS                                   ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define prompts/instructions for your task.            ║
║  Prompts are selected based on task type and returned to the model.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "default": [
        "This is a Tower of Hanoi puzzle with three pegs: Left, Middle (auxiliary), and Right (Goal). Move all disks to the Right peg (Goal). Rules: (1) Move one disk at a time, (2) Only move the top disk of a stack, (3) Never place a larger disk on a smaller disk. You may use the Middle peg as an auxiliary peg. Show the full optimal solution until all disks are stacked on the Goal peg.",
        "Solve the Tower of Hanoi puzzle with three pegs (Left, Middle auxiliary, Right Goal) by moving all disks to the Right peg. Follow the rules: move one disk at a time, only move the top disk, and never place a larger disk on a smaller one. You may use the Middle peg as an auxiliary peg. Demonstrate the full optimal solution to completion.",
        "In this Tower of Hanoi puzzle there are three pegs: Left, Middle (auxiliary), and Right (Goal). Your goal is to move all disks to the Right peg. Remember: one disk at a time, only the top disk can move, and larger disks cannot go on smaller ones. You may use the Middle peg as an auxiliary peg. Show the full optimal sequence of moves until solved.",
    ],
    
    "easy": [
        "This is a Tower of Hanoi puzzle with 3 disks and three pegs: Left, Middle (auxiliary), and Right (Goal). Move all disks to the Right peg (Goal). You may use the Middle peg as an auxiliary peg. Show the full optimal solution following the rules: one disk at a time, only move the top disk, never place larger on smaller.",
    ],
    
    "medium": [
        "This is a Tower of Hanoi puzzle with 4 disks and three pegs: Left, Middle (auxiliary), and Right (Goal). Move all disks to the Right peg (Goal). You may use the Middle peg as an auxiliary peg. Show the full optimal solution following the rules: one disk at a time, only move the top disk, never place larger on smaller.",
    ],
    
    "hard": [
        "This is a Tower of Hanoi puzzle with 5 disks and three pegs: Left, Middle (auxiliary), and Right (Goal). Move all disks to the Right peg (Goal). You may use the Middle peg as an auxiliary peg. Show the full optimal solution following the rules: one disk at a time, only move the top disk, never place larger on smaller.",
    ],
}


def get_prompt(task_type: str = "default") -> str:
    """
    Select a random prompt for the given task type.
    
    Args:
        task_type: Type of task (key in PROMPTS dict)
        
    Returns:
        Random prompt string from the specified type
    """
    prompts = PROMPTS.get(task_type, PROMPTS["default"])
    return random.choice(prompts)


def get_all_prompts(task_type: str = "default") -> list[str]:
    """Get all prompts for a given task type."""
    return PROMPTS.get(task_type, PROMPTS["default"])
