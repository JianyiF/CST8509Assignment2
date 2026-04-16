import gymnasium as gym
from gymnasium import spaces
from swiplserver import PrologMQI
from .screen import Display
import numpy as np


class BlocksWorldEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, size=5):
        super().__init__()

        self.render_mode = render_mode

        # Start Prolog Server
        self.mqi = PrologMQI()
        self.prolog_thread = self.mqi.create_thread()

        # Load blocks_world.pl
        self.prolog_thread.query('[blocks_world]')

        # Build 6-digit State Dictionary
        result = self.prolog_thread.query("state(State)")
        self.states_dict = {item['State']: i for i, item in enumerate(result)}

        # Store keys list for fast lookup
        self.state_keys = list(self.states_dict.keys())   ### UPDATED

        # Build Action Dictionary
        self.actions_dict = {}
        result = self.prolog_thread.query("action(A)")
        # Example: # [{'A': {'functor': 'move', 'args': ['a','b','c']}}]
        for i, A in enumerate(result):
            action_string = A['A']['functor']
            first = True
            # Build string like move(a,b,c)
            for arg in A['A']['args']:
                if first:
                    action_string += '('
                    first = False
                else:
                    action_string += ','
                action_string += str(arg)

            action_string += ')'
            self.actions_dict[i] = action_string

        # Define Observation & Action Spaces
        self.observation_space = spaces.Discrete(len(self.states_dict))
        self.action_space = spaces.Discrete(len(self.actions_dict))

        # Initial State and Target
        self.state = 0
        self.target = 0

        # Display
        if self.render_mode == "human":
            self.display = Display()

    # RESET
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random 6-digit target
        self.target = self.np_random.integers(0, len(self.states_dict))
        self.target_state_3 = self.state_keys[self.target][:3]  # first 3 digits

        if self.render_mode == "human":
            self.display.target = self.target_state_3

        # Reset Prolog world
        self.prolog_thread.query("reset")

        # Get current 3-digit state
        result = self.prolog_thread.query("current_state(State)")
        current_string = result[0]['State']

        # Combine current + target
        full_state = current_string + self.target_state_3   ### UPDATED
        self.state = self.states_dict[full_state]

        return self.state, {}

    # STEP
    def step(self, action):

        # Convert action integer → Prolog action string
        action_string = self.actions_dict[action]

        # Execute action in Prolog
        result = self.prolog_thread.query(f"step({action_string})")

        # Always get current state after attempting action
        result_state = self.prolog_thread.query("current_state(State)")
        current_string = result_state[0]['State']

        # Combine current + target
        full_state = current_string + self.target_state_3
        self.state = self.states_dict[full_state]

        # Reward logic
        if current_string == self.target_state_3:
            reward = 100  # goal reached
            terminated = True
        elif result:  # valid step but not goal
            reward = -1  # default step penalty
            terminated = False
        else:  # invalid or failed step
            reward = -10
            terminated = False

        return self.state, reward, terminated, False, {}

    # RENDER
    def render(self):
        if self.render_mode == "human":
            state_string = self.state_keys[self.state]
            self.display.step(state_string)

    # CLOSE
    def close(self):
        self.mqi.stop()