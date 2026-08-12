import os
import pickle
import random

import numpy as np

from collections import deque # double-ended queue (faster than list for BFS)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']  # These are the only actions our model can take

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"): # If training or no model exists yet
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        weights[-1] = 0.0  # We don't want to place bombs for Task 1
        self.model = weights / weights.sum()
    else: # If testing and model exists
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # If Training: Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob: 
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb. (NOT for Task 1)
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # use this after Task 1 is done and bomb is added
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0]) # changed coz bomb is missing for task 1

    # If Testing or Exploitation: Use model to predict action based on game state
    state = state_to_features(game_state) 
    self.logger.debug(f"Current state: {state}")

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)

def get_bfs_direction(game_state): 
    """
    Function to solve Task 1. 
    Collect all coins as quickly as possible in the field and 
    navigate efficiently to the nearest coin using BFS.
    Finds the shortest walkable path to the nearest coin using
    breadth-first search (BFS) and returns the direction of the
    very first step of that path. Walls and crates are treated as
    non-walkable, so the search automatically routes around them.
                      """         
    field = game_state['field']
    start = game_state['self'][3] # Agent's position
    coins = game_state['coins']
    if not coins: # No coins left to collect
        return 'WAIT'
    coins_set = set(coins) # Convert list of coins to a set, coz faster.
    queue = deque([start]) # BFS queue starts with agent's position
    visited = {start} # To keep track of visited positions and to avoid infinite loops.
    parent = {} # Remembers the parent of each position to reconstruct the path later.
    target = None 
    while queue:
        current = queue.popleft() # Get the next position to explore from the queue and deletes it from the queue.
        if current in coins_set:
            target = current 
            break
        x, y = current # short for: x = current[0], y = current[1]
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (x + dx, y + dy) # Calculate the coordinates of the neighboring position
            if neighbor not in visited and field[neighbor[0]][neighbor[1]] == 0: # Check if the neighbor is walkable and not visited
                visited.add(neighbor) # add to visited set to avoid revisiting it in the future
                parent[neighbor] = current # Remember the parent
                queue.append(neighbor) # Add to queue to explore later
    if target is None:
        return 'WAIT'
    if target == start: # If the agent is already on a coin, just wait
        return 'WAIT'
    step = target
    while parent[step] != start: # Reconstruct the path from the target back to the start
        step = parent[step]
    dx, dy = step[0] - start[0], step[1] - start[1] # Calculate the direction by comparing start and step
    if dx == 1: return 'RIGHT'
    if dx == -1: return 'LEFT'
    if dy == 1: return 'DOWN'
    if dy == -1: return 'UP'


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    return get_bfs_direction(game_state)  # Use BFS to find the direction to the nearest coin


# OLD CODE FROM SAMPLE AGENT, MAYBE NEEDED AGAIN LATER?
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
