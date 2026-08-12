from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features, get_q_values, ACTIONS

ALPHA = 0.1 # learning rate: how much we update our Q-values after each step
GAMMA = 0.9 # discount factor: how much we value future rewards over immediate rewards

def update_q_values(self, old_state, action, reward, new_state):
    """
    Update the Q-values for the given state-action pair based on the received reward and the new state.

    :param self: This object is passed to all callbacks.
    :param old_state: The previous state before taking the action.
    :param action: The action taken in the old state.
    :param reward: The reward received after taking the action.
    :param new_state: The new state after taking the action. (Can be None if the game has ended.)
    """
    old_q_values = get_q_values(self, old_state)
    action_index = ACTIONS.index(action)

    # During Game
    if new_state is not None:
        future_q = np.max(get_q_values(self, new_state)) # Maximum Q-value of all possible actions in new state
    else: # End of Game
        future_q = 0

    td_error = reward + GAMMA * future_q - old_q_values[action_index] # Temporal Difference error
    old_q_values[action_index] += ALPHA * td_error # Update Q-value for the taken action
    self.logger.debug(f"Updated Q-value for state {old_state}, action {action}: {old_q_values[action_index]} (TD Error: {td_error})")

# Events
#PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Called once after setup(). Creates a deque to store amount of entries. At limit: oldest will be deleted, to add new one.


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events)
    update_q_values(self, old_state, self_action, reward, new_state)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    update_q_values(self, last_state, last_action, reward, None)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        #PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
