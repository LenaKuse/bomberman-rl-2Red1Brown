from collections import namedtuple, deque

import pickle
import numpy as np
import os
from typing import List

import events as e
from .callbacks import state_to_features, get_q_values, ACTIONS, get_bfs_target, get_escape_direction, nearest_opponent_position

from settings import BOMB_POWER

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

    return td_error  # Return the TD error for logging and analysis

# Events
MOVED_CLOSER_TO_COIN = "MOVED_CLOSER_TO_COIN"
MOVED_FURTHER_FROM_COIN = "MOVED_FURTHER_FROM_COIN"
NO_PROGRESS_TOWARD_COIN = "NO_PROGRESS_TOWARD_COIN"
MOVED_CLOSER_TO_CRATE = "MOVED_CLOSER_TO_CRATE"
MOVED_FURTHER_FROM_CRATE = "MOVED_FURTHER_FROM_CRATE"
NO_PROGRESS_TOWARD_CRATE = "NO_PROGRESS_TOWARD_CRATE"
MOVED_CLOSER_TO_SAFETY = "MOVED_CLOSER_TO_SAFETY"
MOVED_FURTHER_FROM_SAFETY = "MOVED_FURTHER_FROM_SAFETY"
NO_PROGRESS_TOWARD_SAFETY = "NO_PROGRESS_TOWARD_SAFETY"
MOVED_CLOSER_TO_OPPONENT = "MOVED_CLOSER_TO_OPPONENT"
MOVED_FURTHER_FROM_OPPONENT = "MOVED_FURTHER_FROM_OPPONENT"
NO_PROGRESS_TOWARD_OPPONENT = "NO_PROGRESS_TOWARD_OPPONENT"
PLACED_BOMB_NEAR_OPPONENT = "PLACED_BOMB_NEAR_OPPONENT"

def count_crates_hit(field, position, bomb_power):
    x, y = position
    count = 0
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        for step in range(1, bomb_power + 1):
            nx, ny = x + dx * step, y + dy * step
            if field[nx][ny] == -1:  # Wand stoppt die Explosion
                break
            if field[nx][ny] == 1:  # Kiste
                count += 1
    return count

def opponents_hit_by_bomb(field, position, bomb_power, opponent_positions):
    x, y = position
    count = 0
    for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
        for step in range(1, bomb_power + 1):
            nx, ny = x + dx * step, y + dy * step
            if field[nx][ny] == -1:  # wall stops the blast
                break
            if (nx, ny) in opponent_positions:
                count += 1
    return count

def log_round_start_if_new(self, current_round):
    if current_round != self.last_seen_round:
        self.last_seen_round = current_round
        if self.eval_mode:
            self.eval_round_display_counter += 1
            self.logger.info(f"Start of test round {self.eval_round_display_counter}.")
        else:
            self.train_round_display_counter += 1
            self.logger.info(f"Start of training round {self.train_round_display_counter}.")

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    #self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE) # Called once after setup(). Creates a deque to store amount of entries. At limit: oldest will be deleted, to add new one.

    self.run_name = input("Enter a name for this training run (e.g. 'Lena_A_baseline'): ")
    self.track_progress = input("Do you want to track progress for a plot? (y/n): ").strip().lower() == "y"

    if self.track_progress:
        self.eval_interval = int(input("Evaluate every ... training rounds? (e.g. 50): "))
        self.eval_rounds = int(input("How many rounds per evaluation phase? (e.g. 10): "))
    else:
        self.eval_interval = None
        self.eval_rounds = None

    self.round_reward = 0 # Sum of rewards for the current round
    self.round_td_error = 0 # Sum of absolute TD errors for the current round
    self.round_steps = 0 # Number of steps taken in the current round
    self.train_round_counter = 0 # counts ONLY training rounds, not eval rounds
    self.eval_mode = self.track_progress # Starts with eval phase to record untrained baseline
    self.rounds_since_eval = 0 
    self.eval_counter = 0

    self.last_seen_round = 0
    self.train_round_display_counter = 0
    self.eval_round_display_counter = 0

    os.makedirs("experiments", exist_ok=True)

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
    log_round_start_if_new(self, old_game_state['round'])

    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Compute the escape signal first -- it decides whether coin/crate
    # progress should even be judged this step.
    old_escape_signal, old_dist_to_safety = get_escape_direction(old_game_state)
    new_escape_signal, new_dist_to_safety = get_escape_direction(new_game_state)

    # Additional reward for moving closer to or further away from the nearest coin or/and crate
    # -- only judged while the agent is safe; while fleeing, coin/crate progress is
    # irrelevant and must not be rewarded or punished.
    if old_escape_signal == 'SAFE' and e.COIN_COLLECTED not in events:
        old_target_type, _, old_distance = get_bfs_target(old_game_state)
        new_target_type, _, new_distance = get_bfs_target(new_game_state)
        if old_distance is not None and new_distance is not None:
            if old_target_type == 'COIN':
                if new_distance < old_distance:
                    events.append(MOVED_CLOSER_TO_COIN)
                elif new_distance > old_distance:
                    events.append(MOVED_FURTHER_FROM_COIN)
                else:
                    events.append(NO_PROGRESS_TOWARD_COIN)
            elif old_target_type == 'CRATE':
                if new_distance < old_distance:
                    events.append(MOVED_CLOSER_TO_CRATE)
                elif new_distance > old_distance:
                    events.append(MOVED_FURTHER_FROM_CRATE)
                else:
                    events.append(NO_PROGRESS_TOWARD_CRATE)
            elif old_target_type == 'OPPONENT':
                if new_distance < old_distance:
                    events.append(MOVED_CLOSER_TO_OPPONENT)
                elif new_distance > old_distance:
                    events.append(MOVED_FURTHER_FROM_OPPONENT)
                else:
                    events.append(NO_PROGRESS_TOWARD_OPPONENT)

        # Fallback: no opponent is currently BFS-reachable, so use straight-line
        # distance instead. Lock onto whichever opponent was nearest BEFORE this
        # step, and keep using THEIR OLD position for both distances -- this
        # isolates the agent's own movement from the opponent's, and prevents
        # credit going to a different opponent becoming "nearest" by chance.
        if old_target_type != 'OPPONENT':
            target_opponent_pos = nearest_opponent_position(old_game_state)
            if target_opponent_pos is not None:
                agent_old_pos = old_game_state['self'][3]
                agent_new_pos = new_game_state['self'][3]
                ox, oy = target_opponent_pos
                old_opp_dist = abs(agent_old_pos[0] - ox) + abs(agent_old_pos[1] - oy)
                new_opp_dist = abs(agent_new_pos[0] - ox) + abs(agent_new_pos[1] - oy)

                if new_opp_dist < old_opp_dist:
                    events.append(MOVED_CLOSER_TO_OPPONENT)
                elif new_opp_dist > old_opp_dist:
                    events.append(MOVED_FURTHER_FROM_OPPONENT)
                else:
                    events.append(NO_PROGRESS_TOWARD_OPPONENT)

    #Reward/Punishment for running towards safety
    if old_escape_signal != 'SAFE':
        if new_escape_signal == 'SAFE':
            events.append(MOVED_CLOSER_TO_SAFETY)              # escaped entirely
        elif old_escape_signal == 'TRAPPED' and new_escape_signal == 'TRAPPED':
            events.append(NO_PROGRESS_TOWARD_SAFETY)            # trapped before, trapped still
        elif old_escape_signal == 'TRAPPED':
            events.append(MOVED_CLOSER_TO_SAFETY)               # was trapped, now has an escape route
        elif new_escape_signal == 'TRAPPED':
            events.append(MOVED_FURTHER_FROM_SAFETY)            # had a route, now boxed in
        elif new_dist_to_safety < old_dist_to_safety:
            events.append(MOVED_CLOSER_TO_SAFETY)
        elif new_dist_to_safety > old_dist_to_safety:
            events.append(MOVED_FURTHER_FROM_SAFETY)
        else:
            events.append(NO_PROGRESS_TOWARD_SAFETY)


# Additional reward/punishment for dropping a bomb at good or bad positions            
    bomb_reward = 0
    if self_action == 'BOMB':
        if old_game_state['self'][2]:  # bomb was actually available -> really got placed
            bomb_position = old_game_state['self'][3]
            crates_hit = count_crates_hit(old_game_state['field'], bomb_position, BOMB_POWER)
            opponent_positions = set(o[3] for o in old_game_state['others'])
            opponents_hit = opponents_hit_by_bomb(old_game_state['field'], bomb_position, BOMB_POWER, opponent_positions)

            if opponents_hit > 0:
                events.append(PLACED_BOMB_NEAR_OPPONENT)

            if crates_hit == 0 and opponents_hit == 0:
                bomb_reward = -0.3
            elif crates_hit > 0:
                bomb_reward = crates_hit * 0.3
            else:
                bomb_reward = 0  # hit an opponent but no crates -- not a waste, handled via PLACED_BOMB_NEAR_OPPONENT instead

            self.logger.debug(f"BOMB placed at {bomb_position}, crates_hit={crates_hit}, opponents_hit={opponents_hit}, bomb_reward={bomb_reward}")
        else:
            self.logger.debug("BOMB chosen but no bomb available (invalid action)")

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))
    old_state = state_to_features(old_game_state)
    new_state = state_to_features(new_game_state)
    reward = reward_from_events(self, events) + bomb_reward

    self.round_reward += reward
    self.round_steps += 1

    if not self.eval_mode:  # Only update Q-values during training
        td_error = update_q_values(self, old_state, self_action, reward, new_state) # Update Q-values
        self.round_td_error += abs(td_error) # Add absolute TD error to the round's total for logging purposes

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
    log_round_start_if_new(self, last_game_state['round'])
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    #self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    last_state = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.round_reward += reward # For logging purposes
    self.round_steps += 1 # For logging purposes

    if not self.eval_mode:  # Never during evaluation phase, only during training
        td_error = update_q_values(self, last_state, last_action, reward, None) # Update Q-values
        self.round_td_error += abs(td_error) # Add absolute TD error to the round's total for logging purposes
        avg_td_error = self.round_td_error / self.round_steps if self.round_steps > 0 else 0 # Average TD error for the round, so rounds with different lengths can be compared

        self.train_round_counter += 1 

        if self.track_progress:
            with open(f"experiments/{self.run_name}_training_progress.csv", "a") as f:
                f.write(f"{self.train_round_counter},{self.round_reward},{avg_td_error}\n")

            self.rounds_since_eval += 1
            if self.rounds_since_eval >= self.eval_interval:
                self.eval_mode = True
                self.rounds_since_eval = 0
    else: # During evaluation phase, we don't update Q-values, but we log the evaluation rewards
        if self.track_progress:
            with open(f"experiments/{self.run_name}_eval_progress.csv", "a") as f:
                f.write(f"{self.train_round_counter},{self.round_reward}\n")

        self.eval_counter += 1
        if self.eval_counter >= self.eval_rounds:
            self.eval_mode = False
            self.eval_counter = 0

    self.round_reward = 0 # Reset round reward for the next round
    self.round_td_error = 0 # Reset round TD error for the next round
    self.round_steps = 0 # Reset round steps for the next round

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
        # Coin shaping rewards
        e.COIN_COLLECTED: 1,
        MOVED_CLOSER_TO_COIN: 0.5,
        MOVED_FURTHER_FROM_COIN: -0.5,
        NO_PROGRESS_TOWARD_COIN: -0.2,

        # Crate shaping rewards
        # we do not use the event crate destroyed because it fires at the wrong time!
        MOVED_CLOSER_TO_CRATE: 0.2,
        MOVED_FURTHER_FROM_CRATE: -0.2,
        NO_PROGRESS_TOWARD_CRATE: -0.1,

        #Outrun bombs rewards
        MOVED_CLOSER_TO_SAFETY: 1.0,
        MOVED_FURTHER_FROM_SAFETY: -1.5,
        NO_PROGRESS_TOWARD_SAFETY: -0.3,

        # Other rewards
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -5,
        e.INVALID_ACTION: -1, 

        # Opponent shaping rewards
        MOVED_CLOSER_TO_OPPONENT: 0.8, #before 0.5
        MOVED_FURTHER_FROM_OPPONENT: -1.0, #before -0.5
        NO_PROGRESS_TOWARD_OPPONENT: -0.4, #before -0.3
        PLACED_BOMB_NEAR_OPPONENT: 1.5,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
