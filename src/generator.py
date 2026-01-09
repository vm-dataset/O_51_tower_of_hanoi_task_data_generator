"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK GENERATOR                                 ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to implement your data generation logic.                 ║
║  Replace the example implementation with your own task.                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random
import tempfile
from collections import deque
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from core import BaseGenerator, TaskPair, ImageRenderer
from core.video_utils import VideoGenerator
from .config import TaskConfig
from .prompts import get_prompt

# Goal peg is always the rightmost peg (index 2)
GOAL_PEG = 2


def state_key(state: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """Convert state to hashable tuple for use in dictionaries."""
    return tuple(tuple(p) for p in state)


def get_valid_moves(state: List[List[int]]) -> List[Tuple[int, int, int]]:
    """
    Get all valid moves from current state.
    
    Returns:
        List of (from_peg, to_peg, disk_size) tuples
    """
    moves = []
    state = [list(p) for p in state]
    
    for src in range(3):
        if not state[src]:
            continue
        disk = state[src][-1]  # Top disk
        for dst in range(3):
            if src != dst:
                # Valid if destination is empty or has larger disk on top
                if not state[dst] or state[dst][-1] > disk:
                    moves.append((src, dst, disk))
    
    return moves


def apply_move(state: List[List[int]], move: Tuple[int, int, int]) -> List[List[int]]:
    """Apply a move to the state and return new state."""
    src, dst, _ = move
    new_state = [list(p) for p in state]
    disk = new_state[src].pop()
    new_state[dst].append(disk)
    return new_state


def find_optimal_moves(state: List[List[int]], num_disks: int) -> List[Tuple[int, int, int]]:
    """
    Find all optimal first moves from current state to goal using BFS.
    
    Goal state: all disks on right peg (GOAL_PEG), stacked largest to smallest.
    Uses BFS backwards from goal to compute distance-to-goal for all states.
    """
    # Goal state: all disks on right peg
    goal = tuple(
        tuple(range(num_disks, 0, -1)) if i == GOAL_PEG else ()
        for i in range(3)
    )
    start = state_key(state)
    
    if start == goal:
        return []
    
    # BFS backwards from goal to compute distance-to-goal for all states
    dist_to_goal = {goal: 0}
    queue = deque([goal])
    
    while queue:
        curr = queue.popleft()
        curr_dist = dist_to_goal[curr]
        
        # Try all valid moves from current state
        for move in get_valid_moves(list(curr)):
            nxt = state_key(apply_move(list(curr), move))
            if nxt not in dist_to_goal:
                dist_to_goal[nxt] = curr_dist + 1
                queue.append(nxt)
    
    if start not in dist_to_goal:
        return []  # Unreachable state
    
    # Find all moves that reduce distance by 1
    start_dist = dist_to_goal[start]
    optimal = []
    for move in get_valid_moves(state):
        nxt = state_key(apply_move(state, move))
        if dist_to_goal.get(nxt) == start_dist - 1:
            optimal.append(move)
    
    return optimal


@lru_cache(maxsize=8)
def compute_dist_to_goal(num_disks: int) -> Dict[Tuple[Tuple[int, ...], ...], int]:
    """
    Compute shortest distance-to-goal for all reachable states (BFS backwards from goal).

    State space size is 3^n (<= 243 for n<=5), so this is fast.
    """
    goal = tuple(
        tuple(range(num_disks, 0, -1)) if i == GOAL_PEG else ()
        for i in range(3)
    )
    dist_to_goal: Dict[Tuple[Tuple[int, ...], ...], int] = {goal: 0}
    queue = deque([goal])

    while queue:
        curr = queue.popleft()
        curr_dist = dist_to_goal[curr]
        for move in get_valid_moves(list(curr)):
            nxt = state_key(apply_move(list(curr), move))
            if nxt not in dist_to_goal:
                dist_to_goal[nxt] = curr_dist + 1
                queue.append(nxt)

    return dist_to_goal


def find_optimal_solution_path(state: List[List[int]], num_disks: int) -> List[Tuple[int, int, int]]:
    """
    Return an optimal move sequence from `state` to goal.

    Uses precomputed distance-to-goal and greedily takes any move that reduces distance by 1.
    """
    dist_to_goal = compute_dist_to_goal(num_disks)
    curr = [list(p) for p in state]
    curr_key = state_key(curr)

    if curr_key not in dist_to_goal:
        return []

    moves: List[Tuple[int, int, int]] = []
    while True:
        curr_key = state_key(curr)
        curr_dist = dist_to_goal.get(curr_key)
        if curr_dist is None or curr_dist == 0:
            break

        candidates = []
        for move in get_valid_moves(curr):
            nxt = apply_move(curr, move)
            nxt_key = state_key(nxt)
            if dist_to_goal.get(nxt_key) == curr_dist - 1:
                candidates.append(move)

        if not candidates:
            break

        chosen = random.choice(candidates)
        moves.append(chosen)
        curr = apply_move(curr, chosen)

    return moves


def generate_random_valid_state(num_disks: int) -> List[List[int]]:
    """
    Generate a random valid state that isn't already solved.
    
    Places disks randomly on pegs while respecting the rules.
    """
    goal = [[], [], list(range(num_disks, 0, -1))]
    
    while True:
        state = [[], [], []]
        # Place disks from largest to smallest
        for disk in range(num_disks, 0, -1):
            # Find valid pegs (empty or with larger disk on top)
            valid_pegs = [p for p in range(3) 
                         if not state[p] or state[p][-1] > disk]
            state[random.choice(valid_pegs)].append(disk)
        
        if state != goal:
            return state


class TaskGenerator(BaseGenerator):
    """
    Tower of Hanoi task generator.
    
    Generates FULL-SOLUTION reasoning tasks where the model must show
    the full optimal sequence of moves to solve the puzzle.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.renderer = ImageRenderer(image_size=config.image_size)
        
        # Initialize video generator if enabled
        self.video_generator = None
        if config.generate_videos and VideoGenerator.is_available():
            self.video_generator = VideoGenerator(
                fps=config.video_fps, 
                output_format="mp4"
            )

        # Small LRU cache for rendered static states (speed-up for full-solution videos)
        # Key: (num_disks, state_key)
        self._state_img_cache: OrderedDict[tuple[int, tuple[tuple[int, ...], ...]], Image.Image] = OrderedDict()
        self._state_img_cache_max = 256
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        """Generate one Tower of Hanoi task pair."""
        
        # Determine number of disks based on difficulty
        num_disks = self._select_num_disks()
        difficulty = self._get_difficulty_name(num_disks)
        
        # Generate initial state
        initial_state = generate_random_valid_state(num_disks)

        # Find optimal FULL solution path to goal
        solution_moves = find_optimal_solution_path(initial_state, num_disks)
        if not solution_moves:
            # Shouldn't happen for valid states; retry
            return self.generate_task_pair(task_id)

        # Apply all moves to reach goal
        final_state = [list(p) for p in initial_state]
        for mv in solution_moves:
            final_state = apply_move(final_state, mv)
        
        # Render images
        first_image = self._render_hanoi_state(initial_state, num_disks)
        final_image = self._render_hanoi_state(final_state, num_disks)
        
        # Generate video (optional)
        video_path = None
        if self.config.generate_videos and self.video_generator:
            video_path = self._generate_video(
                initial_state, solution_moves, num_disks, task_id
            )
        
        # Select prompt based on difficulty (full-solution prompt)
        prompt = get_prompt(difficulty)
        
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=prompt,
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path
        )

    def generate_dataset(self) -> List[TaskPair]:
        """
        Generate dataset, optionally enforcing uniqueness.

        Uniqueness is defined by (num_disks, initial_state). This guarantees that within a single run
        you will not get duplicate puzzles (same starting configuration with same disk count).
        """
        pairs: List[TaskPair] = []
        ensure_unique = getattr(self.config, "ensure_unique", True)
        max_attempts = int(getattr(self.config, "max_unique_attempts_per_task", 500))

        seen: set[tuple[int, tuple[tuple[int, ...], ...]]] = set()

        for i in range(self.config.num_samples):
            task_id = f"{self.config.domain}_{i:04d}"

            if not ensure_unique:
                pair = self.generate_task_pair(task_id)
                pairs.append(pair)
                print(f"  Generated: {task_id}")
                continue

            # Resample until we get a fresh initial state (cheap part first)
            created = False
            for attempt in range(max_attempts):
                num_disks = self._select_num_disks()
                initial_state = generate_random_valid_state(num_disks)
                sig = (num_disks, state_key(initial_state))
                if sig in seen:
                    continue

                # Mark as seen before heavier rendering/video work
                seen.add(sig)

                # Now build the full task from this fixed initial_state
                # (inline the logic from generate_task_pair to avoid regenerating a different state)
                difficulty = self._get_difficulty_name(num_disks)
                solution_moves = find_optimal_solution_path(initial_state, num_disks)
                if not solution_moves:
                    # extremely unlikely; free this signature and retry
                    seen.remove(sig)
                    continue

                final_state = [list(p) for p in initial_state]
                for mv in solution_moves:
                    final_state = apply_move(final_state, mv)

                first_image = self._render_hanoi_state(initial_state, num_disks)
                final_image = self._render_hanoi_state(final_state, num_disks)

                video_path = None
                if self.config.generate_videos and self.video_generator:
                    video_path = self._generate_video(
                        initial_state,
                        solution_moves,
                        num_disks,
                        task_id,
                    )

                prompt = get_prompt(difficulty)
                pair = TaskPair(
                    task_id=task_id,
                    domain=self.config.domain,
                    prompt=prompt,
                    first_image=first_image,
                    final_image=final_image,
                    ground_truth_video=video_path,
                )

                pairs.append(pair)
                print(f"  Generated: {task_id}")
                created = True
                break

            if not created:
                raise RuntimeError(
                    f"Failed to generate a unique task after {max_attempts} attempts "
                    f"(i={i}, domain={self.config.domain}). Consider increasing max_unique_attempts_per_task "
                    f"or reducing num_samples / disk range."
                )

        return pairs
    
    def _select_num_disks(self) -> int:
        """Select number of disks based on config."""
        if hasattr(self.config, 'num_disks') and self.config.num_disks:
            return self.config.num_disks
        
        # Random selection based on difficulty distribution
        if hasattr(self.config, 'min_disks') and hasattr(self.config, 'max_disks'):
            return random.randint(self.config.min_disks, self.config.max_disks)
        
        # Default: cycle through difficulties
        difficulties = [3, 4, 5]  # easy, medium, hard
        return random.choice(difficulties)
    
    def _get_difficulty_name(self, num_disks: int) -> str:
        """Get difficulty name based on number of disks."""
        if num_disks == 3:
            return "easy"
        elif num_disks == 4:
            return "medium"
        elif num_disks >= 5:
            return "hard"
        return "default"
    
    def _render_hanoi_state(
        self, 
        state: List[List[int]], 
        num_disks: int
    ) -> 'Image.Image':
        """
        Render Tower of Hanoi state as an image.
        
        Uses matplotlib to create a visualization with 3 pegs and disks.
        """
        cache_key = (num_disks, state_key(state))
        cached = self._state_img_cache.get(cache_key)
        if cached is not None:
            # refresh LRU
            self._state_img_cache.move_to_end(cache_key)
            return cached

        width, height = self.config.image_size
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Disk colors (cycling if more than 4 disks)
        disk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#A8E6CF']
        
        # Peg positions (left, middle, right)
        peg_x = [width * 0.25, width * 0.5, width * 0.75]
        peg_labels = ['Left', 'Middle (Aux)', 'Right (Goal)']
        
        # Base platform
        base_height = height * 0.1
        ax.add_patch(Rectangle(
            (width * 0.05, 0), 
            width * 0.9, 
            base_height, 
            facecolor='#8B4513',
            edgecolor='none',
            linewidth=0
        ))
        
        # Pegs
        peg_width = width * 0.02
        peg_height = height * 0.6
        for x in peg_x:
            ax.add_patch(Rectangle(
                (x - peg_width/2, base_height),
                peg_width,
                peg_height,
                facecolor='#A0522D',
                edgecolor='none',
                linewidth=0
            ))
        
        # Disks
        disk_height = height * 0.08
        max_disk_width = width * 0.25
        
        for peg_idx, peg in enumerate(state):
            for disk_idx, disk in enumerate(peg):
                # Disk width scales with size
                disk_width = max_disk_width * (disk / num_disks)
                x = peg_x[peg_idx] - disk_width / 2
                y = base_height + disk_idx * disk_height
                color = disk_colors[(disk - 1) % len(disk_colors)]
                
                ax.add_patch(Rectangle(
                    (x, y),
                    disk_width,
                    disk_height * 0.9,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                ))
        
        # Goal highlight (green dashed box around right peg)
        goal_box_width = width * 0.3
        goal_box_height = height * 0.7
        ax.add_patch(Rectangle(
            (peg_x[2] - goal_box_width/2, base_height),
            goal_box_width,
            goal_box_height,
            fill=False,
            edgecolor='green',
            linewidth=3,
            linestyle='--'
        ))
        
        # Labels
        label_y = -height * 0.05
        for i, (x, label) in enumerate(zip(peg_x, peg_labels)):
            color = 'green' if i == 2 else 'black'
            weight = 'bold' if i == 2 else 'normal'
            ax.text(
                x, label_y, label,
                ha='center', va='top',
                fontsize=max(10, width // 50),
                color=color,
                fontweight=weight
            )
        
        ax.set_xlim(0, width)
        ax.set_ylim(-height * 0.1, height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Convert to PIL Image
        fig.canvas.draw()
        # Get the buffer as numpy array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        buf = buf.reshape((height, width, 4))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        plt.close(fig)
        
        out = ImageRenderer.ensure_rgb(Image.fromarray(buf))

        # Update cache (LRU)
        self._state_img_cache[cache_key] = out
        self._state_img_cache.move_to_end(cache_key)
        while len(self._state_img_cache) > self._state_img_cache_max:
            self._state_img_cache.popitem(last=False)

        return out
    
    def _generate_video(
        self,
        initial_state: List[List[int]],
        moves: List[Tuple[int, int, int]],
        num_disks: int,
        task_id: str
    ) -> Optional[str]:
        """
        Generate ground truth video showing disk movement.
        
        Creates smooth animation of the full optimal solution (multiple moves).
        """
        temp_dir = Path(tempfile.gettempdir()) / f"{self.config.domain}_videos"
        temp_dir.mkdir(parents=True, exist_ok=True)
        video_path = temp_dir / f"{task_id}_ground_truth.mp4"
        
        # Create animation frames
        frames = self._create_hanoi_animation_frames(
            initial_state,
            moves,
            num_disks,
            hold_frames=getattr(self.config, "hold_frames", 2),
            transition_frames=getattr(self.config, "transition_frames", 8),
        )
        
        if not frames:
            return None
        
        result = self.video_generator.create_video_from_frames(frames, video_path)
        return str(result) if result else None
    
    def _create_hanoi_animation_frames(
        self,
        initial_state: List[List[int]],
        moves: List[Tuple[int, int, int]],
        num_disks: int,
        hold_frames: int = 5,
        transition_frames: int = 25
    ) -> List['Image.Image']:
        """
        Create animation frames showing disk movement.
        
        The moving disk slides smoothly for each move in the full solution sequence.
        """
        frames = []

        # Hold initial position once
        curr_state = [list(p) for p in initial_state]
        curr_image = self._render_hanoi_state(curr_state, num_disks)
        for _ in range(hold_frames):
            frames.append(curr_image)

        for mv in moves:
            nxt_state = apply_move(curr_state, mv)
            frames.extend(
                self._create_single_move_frames(
                    curr_state, nxt_state, mv, num_disks,
                    hold_frames=1,
                    transition_frames=transition_frames
                )
            )
            curr_state = nxt_state

        # Hold final solved state
        final_image = self._render_hanoi_state(curr_state, num_disks)
        for _ in range(hold_frames):
            frames.append(final_image)

        return frames

    def _create_single_move_frames(
        self,
        state_before: List[List[int]],
        state_after: List[List[int]],
        move: Tuple[int, int, int],
        num_disks: int,
        hold_frames: int = 1,
        transition_frames: int = 25,
    ) -> List['Image.Image']:
        """Create frames for a single disk move (used to build full-solution videos)."""
        frames: List['Image.Image'] = []
        src_peg, dst_peg, disk_size = move

        # Hold start state briefly
        start_image = self._render_hanoi_state(state_before, num_disks)
        for _ in range(hold_frames):
            frames.append(start_image)

        width, height = self.config.image_size
        peg_x = [width * 0.25, width * 0.5, width * 0.75]
        base_height = height * 0.1

        src_x = peg_x[src_peg]
        dst_x = peg_x[dst_peg]

        # Y positions computed from current stacks BEFORE the move
        src_stack_height = len(state_before[src_peg]) - 1
        dst_stack_height = len(state_before[dst_peg])

        disk_height = height * 0.08
        src_y = base_height + src_stack_height * disk_height
        dst_y = base_height + dst_stack_height * disk_height

        max_disk_width = width * 0.25
        disk_width = max_disk_width * (disk_size / num_disks)
        disk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#A8E6CF']
        disk_color = disk_colors[(disk_size - 1) % len(disk_colors)]

        for i in range(transition_frames):
            progress = i / (transition_frames - 1) if transition_frames > 1 else 1.0
            if progress >= 1.0:
                frames.append(self._render_hanoi_state(state_after, num_disks))
            else:
                frames.append(
                    self._render_hanoi_state_with_moving_disk(
                        state_before,
                        move,
                        num_disks,
                        src_x + (dst_x - src_x) * progress,
                        src_y + (dst_y - src_y) * progress,
                        disk_width,
                        disk_height,
                        disk_color,
                    )
                )

        return frames
    
    def _render_hanoi_state_with_moving_disk(
        self,
        state: List[List[int]],
        move: Tuple[int, int, int],
        num_disks: int,
        disk_x: float,
        disk_y: float,
        disk_width: float,
        disk_height: float,
        disk_color: str
    ) -> 'Image.Image':
        """
        Render state with a disk at an intermediate position.
        
        Used for animation frames where the disk is in transit.
        """
        width, height = self.config.image_size
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        disk_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFE66D', '#A8E6CF']
        peg_x = [width * 0.25, width * 0.5, width * 0.75]
        
        # Base
        base_height = height * 0.1
        ax.add_patch(Rectangle(
            (width * 0.05, 0),
            width * 0.9,
            base_height,
            facecolor='#8B4513',
            edgecolor='none',
            linewidth=0
        ))
        
        # Pegs
        peg_width = width * 0.02
        peg_height = height * 0.6
        for x in peg_x:
            ax.add_patch(Rectangle(
                (x - peg_width/2, base_height),
                peg_width,
                peg_height,
                facecolor='#A0522D',
                edgecolor='none',
                linewidth=0
            ))
        
        # Render state without the moving disk
        src_peg, _, disk_size = move
        state_without_moving = [list(p) for p in state]
        state_without_moving[src_peg] = state_without_moving[src_peg][:-1]  # Remove top disk
        
        max_disk_width = width * 0.25
        disk_height_unit = height * 0.08
        
        for peg_idx, peg in enumerate(state_without_moving):
            for disk_idx, disk in enumerate(peg):
                d_width = max_disk_width * (disk / num_disks)
                x = peg_x[peg_idx] - d_width / 2
                y = base_height + disk_idx * disk_height_unit
                color = disk_colors[(disk - 1) % len(disk_colors)]
                
                ax.add_patch(Rectangle(
                    (x, y),
                    d_width,
                    disk_height_unit * 0.9,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=2
                ))
        
        # Render moving disk at intermediate position
        ax.add_patch(Rectangle(
            (disk_x - disk_width / 2, disk_y),
            disk_width,
            disk_height * 0.9,
            facecolor=disk_color,
            edgecolor='black',
            linewidth=2
        ))
        
        # Goal highlight
        goal_box_width = width * 0.3
        goal_box_height = height * 0.7
        ax.add_patch(Rectangle(
            (peg_x[2] - goal_box_width/2, base_height),
            goal_box_width,
            goal_box_height,
            fill=False,
            edgecolor='green',
            linewidth=3,
            linestyle='--'
        ))
        
        # Labels
        label_y = -height * 0.05
        peg_labels = ['Left', 'Middle (Aux)', 'Right (Goal)']
        for i, (x, label) in enumerate(zip(peg_x, peg_labels)):
            color = 'green' if i == 2 else 'black'
            weight = 'bold' if i == 2 else 'normal'
            ax.text(
                x, label_y, label,
                ha='center', va='top',
                fontsize=max(10, width // 50),
                color=color,
                fontweight=weight
            )
        
        ax.set_xlim(0, width)
        ax.set_ylim(-height * 0.1, height)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Convert to PIL Image
        fig.canvas.draw()
        # Get the buffer as numpy array
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        width, height = fig.canvas.get_width_height()
        buf = buf.reshape((height, width, 4))
        # Convert RGBA to RGB
        buf = buf[:, :, :3]
        plt.close(fig)
        
        return ImageRenderer.ensure_rgb(Image.fromarray(buf))
