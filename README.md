# Bomberman RL — <Your Team Name>

**Final project for Machine Learning Essentials (Summer Semester 2026).**    
Team: *Sunna Gottschewski*, *Mohamed Attia*, *Lena Kuse*

## Overview

We train reinforcement learning agents to play Bomberman. Our repository contains two agent implementations (at least one of the models focuses on techniques from the lecture):

- `agent_code/model_a/` — *!-- TODO: describe model A here once decided -->one-line description, e.g. "tabular Q-learning
  with hand-crafted features"*
- `agent_code/model_b/` — *!-- TODO: describe model B here once decided -->one-line description, e.g. "CNN-based DQN"*

## Setup
We use the conda environment ml_homework from the lecture.  
A standard Python 3 installation with numpy, spicy and sklearn is required.  
Additional packages to be installed, according to the task:  
```bash
conda activate ml_homework
pip install pygame tqdm
pip install -r requirements.txt
```
*Note: The requirements.txt contains all additional packages needed for our agent to function.   
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

```bash
python main.py play --my-agent model_a --train 1 --no-gui
```

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

- Agent code: Mon 21.09.2026, 21:00
- Report: Mon 28.09.2026, 21:00
