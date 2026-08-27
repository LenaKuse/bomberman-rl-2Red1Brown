import os
import pickle
import random

import numpy as np

from collections import deque # double-ended queue (faster than list for BFS)


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']  # These are the only actions our model can take

def get_q_values(self, state):
    """
    Get the Q-values for a given state from the model.

    :param self: This agent's persistent object. 
    :param state: The current state of the game.
    :return: A numpy array of Q-values for each action.
    """
    if state not in self.model:
        self.model[state] = np.zeros(len(ACTIONS)) # Initialize Q-values for yet unseen states
    return self.model[state]
    
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
        self.model = {} # Initialize an empty dictionary to store Q-values for each state
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
    # If Training and no evaluation phase: Exploration vs exploitation
    random_prob = .1
    if self.train and not getattr(self, 'eval_mode', False) and random.random() < random_prob: 
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb. (NOT for Task 1)
        # return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) # use this after Task 1 is done and bomb is added
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .2, .0]) # changed coz bomb is missing for task 1

    # If Testing or Exploitation: Use model to predict action based on game state
    state = state_to_features(game_state) 
    q_values = get_q_values(self, state) # Get Q-values for the current state

    masked_q_values = q_values.copy() # Create a copy of Q-values to mask invalid actions
    masked_q_values[-1] = -np.inf # Exclude 'BOMB' action for Task 1

    # Choose the action with the highest Q-value, breaking ties randomly.
    best_value = np.max(masked_q_values)
    best_indices = np.flatnonzero(masked_q_values == best_value)
    action = ACTIONS[np.random.choice(best_indices)]
    self.logger.debug(f"State: {state} (Q-values: {q_values}) -> Chosen action: {action}")
    return action

def get_bfs_target(game_state):
    """
    Single BFS search that finds BOTH the nearest coin and the nearest
    free tile adjacent to a crate (i.e. a spot to bomb from).
    Coins win ties (same distance).

    :param game_state: A dictionary describing the current game board.
    :return: A tuple (target_type, direction, distance).
             target_type is 'COIN', 'CRATE', or None (nothing found).
             direction is one of ACTIONS (UP, RIGHT, DOWN, LEFT), 'AT_TARGET' or 'WAIT' (no target found).
    """
    if game_state is None:
        return None, 'WAIT', None

    field = game_state['field']
    start = game_state['self'][3]
    coins_set = set(game_state['coins'])

    if start in coins_set:
        return 'COIN', 'AT_TARGET', 0

    queue = deque([start])
    visited = {start}
    parent = {}
    dist = {start: 0}

    nearest_coin = None
    nearest_coin_dist = None
    nearest_crate_spot = None
    nearest_crate_dist = None

    while queue:
        # BFS explores in order of increasing distance, so once we've
        # found one of each type, nothing further in the queue can beat it
        if nearest_coin is not None and nearest_crate_spot is not None:
            break

        current = queue.popleft()
        cur_dist = dist[current]
        x, y = current

        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            neighbor = (x + dx, y + dy)
            if neighbor in visited:
                continue
            nx, ny = neighbor

            if field[nx][ny] == 0:  # free, walkable tile
                visited.add(neighbor)
                parent[neighbor] = current
                dist[neighbor] = cur_dist + 1
                queue.append(neighbor)

                if neighbor in coins_set and nearest_coin is None:
                    nearest_coin = neighbor
                    nearest_coin_dist = dist[neighbor]

            elif field[nx][ny] == 1:  # crate -- can't walk onto it
                if nearest_crate_spot is None:
                    nearest_crate_spot = current  # bomb from HERE
                    nearest_crate_dist = cur_dist
                visited.add(neighbor)  # don't re-discover the same crate

    # Decide which target wins: coin wins ties
    if nearest_coin is not None and (
        nearest_crate_spot is None or nearest_coin_dist <= nearest_crate_dist
    ):
        target, target_type, distance = nearest_coin, 'COIN', nearest_coin_dist
    elif nearest_crate_spot is not None:
        target, target_type, distance = nearest_crate_spot, 'CRATE', nearest_crate_dist
    else:
        return None, 'WAIT', None

    if target == start:
        return target_type, 'AT_TARGET', 0

    step = target
    while parent[step] != start:
        step = parent[step]
    dx, dy = step[0] - start[0], step[1] - start[1]
    if dx == 1: direction = 'RIGHT'
    elif dx == -1: direction = 'LEFT'
    elif dy == 1: direction = 'DOWN'
    elif dy == -1: direction = 'UP'

    return target_type, direction, distance   


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
    target_type, direction, distance = get_bfs_target(game_state)
    return target_type, direction  # Use BFS to find the direction to the nearest coin


# OLD CODE FROM SAMPLE AGENT, MAYBE NEEDED AGAIN LATER?
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector
    #return stacked_channels.reshape(-1)
