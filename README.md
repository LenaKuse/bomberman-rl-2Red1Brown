# Bomberman RL — <Your Team Name>

**Final project for Machine Learning Essentials (Summer Semester 2026).**    
Team: *Sunna Gottschewski*, *Mohamed Attia*, *Lena Kuse*

## Overview

We train reinforcement learning agents to play Bomberman. Our repository contains two agent implementations (at least one of the models focuses on techniques from the lecture):

- `agent_code/model_a/` — <!-- TODO: describe model A here once decided -->one-line description, e.g. "tabular Q-learning
  with hand-crafted features"*
- `agent_code/model_b/` — <!-- TODO: describe model B here once decided -->one-line description, e.g. "CNN-based DQN"*

## Setup
We use the conda environment ml_homework from the lecture.  
A standard Python 3 installation with numpy, spicy and sklearn is required.  
Additional packages to be installed, according to the task:  
```bash
conda activate ml_homework
pip install pygame tqdm
pip install -r requirements.txt
```

*Note: All of the packages stated above are included in the DOCKERFILE and will be installed. The requirements.txt contains all additional packages needed for OUR agent to function.   
NOTE FOR THE TEAM: If you want to use libraries that aren't installed in the Dockerfile by default, you must specify them in the requirements.txt file.*

# Shortcuts for the Team:
## Running an agent
Run your agent. It plays against three strong rule-based agents.
```bash
python main.py play --my-agent model_a
```
If you want to specify more than one agent to run, call
```bash
python main.py play --agents model_a random_agent rule_based_agent peaceful_agent
```

## Training an agent

### Training Progress Tracking & Plots

When starting training (`--train 1`), you'll be asked four questions:

| Question | What it means |
|---|---|
| Run name | Used in output filenames, so different runs don't overwrite each other (e.g. `Lena_A_baseline`) |
| Track progress? (y/n) | If `n`, training runs normally with no extra logging or evaluation phases |
| Evaluate every ... training rounds? | How many *training* rounds happen between evaluation phases (e.g. `50`) |
| Rounds per evaluation phase? | How many rounds each evaluation phase lasts (e.g. `10`) |

If you say `y`, an **evaluation phase runs first, before any training** — this records the
untrained baseline. After that, training and evaluation alternate based on the two numbers above.
During evaluation, the agent acts greedily (no exploration) and the Q-table is **not** updated.

#### How many total rounds to pass to `--n-rounds`

`--n-rounds` sets the *total* number of rounds (training + evaluation combined) — it does **not**
equal the number of training rounds you'll actually get. Use this formula to compute it:

>'''
>n_rounds = eval_rounds + k*(eval_interval + eval_rounds)
>'''
where `k` = how many full training/eval cycles you want. Choosing `n_rounds` this way makes the
run end right after a completed evaluation phase (so you get a clean final data point).

**Example:** `eval_interval=50`, `eval_rounds=10`, want `k=10` cycles (~500 training rounds)
→ `n_rounds = 10 + 10 × 60 = 610`

#### Generating the plot

```bash
python agent_code/model_a/plot_progress.py <run_name>
```

Creates `agent_code/model_a/experiments/<run_name>_progress.png`. Requires at least
`MIN_ROUNDS_FOR_PLOT` (currently 20) completed training rounds — shorter runs are skipped
with a message instead of producing a misleading plot.

#### What the plots show

- **Training reward** (top, blue): reward per training round. Faint = raw values, thick = smoothed
  trend (centered rolling average — averages nearby rounds on *both* sides, not just before).
- **Eval reward** (top, red): mean ± standard deviation of reward across each evaluation phase,
  plotted at the number of training rounds completed *so far*. The x=0 point is the untrained
  baseline. This is the key curve for checking generalization — if training reward rises but eval
  reward doesn't, that's a sign of overfitting to specific training rounds.
- **Avg. TD-error** (bottom): how much the Q-values are still changing per update. Trending toward
  0 suggests the Q-table is stabilizing (though with our small state space it likely won't reach
  exactly 0 — see report discussion on state aliasing).

#### Changing the smoothing window

The window size is set directly in `plot_progress.py`:
```python
window = 11
```
This averages 5 rounds before and 5 after each point. Increase it for a smoother (but more
delayed/blurred) trend line, decrease it to see more short-term detail.

## Repository structure

- `agent_code/` — one self-contained subdirectory per agent (framework
  requirement: each folder must run independently, no shared imports)
- `shared/` — our own evaluation/plotting scripts, not used by the agents
  themselves
- `experiments/` — training logs, result tables, plots for the report

## Team workflow

- Branch naming: `model-a/<feature>`, `model-b/<feature>`
- Open a pull request into `model-a` or `model-b`; one teammate reviews before merge
- `main` is always kept in a working state, so we can always just zip up main and hand it in without crashing.
- Only merge `model-a`/`model-b` into `main` once they are tested!

## Deadlines

- Agent code test deadline:  Thur 17.09.26, 21:00
- Agent code: Mon 21.09.2026, 21:00
- Report: Mon 28.09.2026, 21:00
