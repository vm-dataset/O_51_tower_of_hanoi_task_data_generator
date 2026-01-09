# Tower of Hanoi Task Data Generator 🎯

A data generator for creating Tower of Hanoi reasoning tasks. Generates full-solution reasoning tasks where video models must demonstrate the full optimal sequence of moves to solve the puzzle.

Repository: [O_51_tower_of_hanoi_task_data_generator](https://github.com/vm-dataset/O_51_tower_of_hanoi_task_data_generator)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vm-dataset/O_51_tower_of_hanoi_task_data_generator.git
cd O_51_tower_of_hanoi_task_data_generator

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Generate tasks
python examples/generate.py --num-samples 50
```

---

## 📁 Structure

```
tower-of-hanoi-task-data-generator/
├── core/                    # ✅ KEEP: Standard utilities
│   ├── base_generator.py   # Abstract base class
│   ├── schemas.py          # Pydantic models
│   ├── image_utils.py      # Image helpers
│   ├── video_utils.py      # Video generation
│   └── output_writer.py    # File output
├── src/                     # ⚠️ CUSTOMIZE: Tower of Hanoi task logic
│   ├── generator.py        # Tower of Hanoi generator
│   ├── prompts.py          # Task prompt templates
│   └── config.py           # Task configuration
├── examples/
│   └── generate.py         # Entry point
└── data/questions/         # Generated output
```

---

## 📦 Output Format

Every generator produces:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state (REQUIRED)
├── final_frame.png          # Goal state (all disks on the Goal peg)
├── prompt.txt               # Instructions (REQUIRED)
└── ground_truth.mp4         # Solution video (OPTIONAL)
```

---

## 🎨 Customization (3 Files to Modify)

### 1. Update `src/generator.py`

The Tower of Hanoi generator implements full-solution reasoning tasks:

```python
from core import BaseGenerator, TaskPair, ImageRenderer

class TaskGenerator(BaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.renderer = ImageRenderer(config.image_size)
    
    def generate_task_pair(self, task_id: str) -> TaskPair:
        # 1. Generate your problem
        initial_state = generate_random_valid_state(num_disks)
        
        # 2. Solve it
        solution_moves = find_optimal_solution_path(initial_state, num_disks)
        
        # 3. Render images
        first_image = self._render_hanoi_state(initial_state, num_disks)
        final_image = self._render_hanoi_state(final_state, num_disks)
        
        # 4. Create TaskPair
        return TaskPair(
            task_id=task_id,
            domain=self.config.domain,
            prompt=self.select_prompt(),
            first_image=first_image,
            final_image=final_image,
            ground_truth_video=video_path  # Optional
        )
```

### 2. Update `src/prompts.py`

Tower of Hanoi prompts for different difficulty levels:

```python
PROMPTS = {
    "easy": [
        "Show the full optimal solution to move all 3 disks to the Goal peg.",
        "Demonstrate the complete sequence of moves to solve the Tower of Hanoi puzzle with 3 disks.",
    ],
    "medium": [
        "Show the full optimal solution to move all 4 disks to the Goal peg.",
        "Demonstrate the complete sequence of moves to solve the Tower of Hanoi puzzle with 4 disks.",
    ],
    "hard": [
        "Show the full optimal solution to move all 5 disks to the Goal peg.",
        "Demonstrate the complete sequence of moves to solve the Tower of Hanoi puzzle with 5 disks.",
    ],
}

def get_prompt(difficulty: str = "default") -> str:
    prompts = PROMPTS.get(difficulty, PROMPTS["easy"])
    return random.choice(prompts)
```

### 3. Update `src/config.py`

**All hyperparameters go here** - both general and task-specific:

```python
from core import GenerationConfig
from pydantic import Field

class TaskConfig(GenerationConfig):
    """Tower of Hanoi task-specific configuration."""
    # Inherits: num_samples, domain, seed, output_dir, image_size
    
    # Override defaults
    domain: str = Field(default="tower_of_hanoi")
    image_size: tuple[int, int] = Field(default=(512, 512))
    
    # Task-specific hyperparameters
    num_disks: Optional[int] = Field(default=None, description="Fixed number of disks (3-5)")
    min_disks: int = Field(default=3, description="Minimum disks for random generation")
    max_disks: int = Field(default=5, description="Maximum disks for random generation")
    difficulty_distribution: dict[str, float] = Field(
        default={"easy": 0.4, "medium": 0.4, "hard": 0.2}
    )
```

**Single entry point:** `python examples/generate.py --num-samples 50`

---

## 🔧 Usage Examples

```bash
# Generate 50 tasks (default settings)
python examples/generate.py --num-samples 50

# Generate without videos (faster)
python examples/generate.py --num-samples 50 --no-videos

# Scale up: generate many tasks with smaller videos
python examples/generate.py --num-samples 5000 --min-disks 3 --max-disks 5 --video-fps 10 --hold-frames 1 --transition-frames 4 --output data/questions_v2

# Scale up (fastest): generate images + prompts only, no MP4
python examples/generate.py --num-samples 20000 --min-disks 3 --max-disks 5 --no-videos --output data/questions_v2

# Custom output directory and seed
python examples/generate.py --num-samples 100 --output data/my_tasks --seed 42
```

---

## 🎯 Task Description

**Tower of Hanoi Full-Solution Reasoning Task**

Each task consists of:
- **Initial State**: A valid Tower of Hanoi configuration with 3-5 disks on 3 pegs
- **Goal**: Move all disks to the right peg (Goal peg)
- **Task**: Show the full optimal solution until all disks are stacked on the Goal peg

**Rules**:
1. Move one disk at a time
2. Only move the top disk of a stack
3. Never place a larger disk on a smaller disk

**Difficulty Levels**:
- **Easy**: 3 disks (27 possible states)
- **Medium**: 4 disks (81 possible states)
- **Hard**: 5 disks (243 possible states)

**Algorithm**: Uses BFS (Breadth-First Search) backwards from goal state to compute optimal moves.
