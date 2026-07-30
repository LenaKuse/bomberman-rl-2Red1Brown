# Bomberman RL — <Your Team Name>

Final project for Machine Learning Essentials (Summer Semester 2026).
Team: <Sunna Gottschewski>, <Mohamed Attia>, <Lena Kuse>

## Overview

We train reinforcement learning agents to play Bomberman. Our repository contains two agent implementations:

- `agent_code/model_a/` — <one-line description, e.g. "tabular Q-learning
  with hand-crafted features">
- `agent_code/model_b/` — <one-line description, e.g. "CNN-based DQN">

## Setup
The following additional libraries need to be installed.
```bash
conda activate ml_homework
pip install -r requirements.txt
```

## Running an agent

```bash
python main.py play --my-agent model_a
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
- Open a PR into `model-a` or `model-b`; one teammate reviews before merge
- `main` is always kept in a working state

## Deadlines

- Agent code: Mon 21.09.2026, 21:00
- Report: Mon 28.09.2026, 21:00