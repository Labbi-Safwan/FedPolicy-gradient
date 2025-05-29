import numpy as np
import gym
from gym import spaces


from .utils import RandomSimplexVector

class FiniteStateFiniteActionMDP(object):
    def __init__(self, S=50, A=50, epsilon_p = 0., epsilon_r = 0., common_transition=None, common_reward=None):
        super().__init__()
        self.S = S
        self.A = A
        self.common_transition = common_transition
        self.common_reward = common_reward
        self.epsilon_p = epsilon_p
        self.epsilon_r = epsilon_r

        # transition kernel
        if self.common_transition is not None:
            self.local_P = RandomSimplexVector(d = self.S, size=[self.S, self.A])
            self.P = (1 - self.epsilon_p)*self.common_transition + self.epsilon_p*self.local_P
        else: self.P = RandomSimplexVector(d = self.S, size=[self.S, self.A])
        
        if self.common_reward is not None:
            self.local_R = np.random.uniform(0., 1., size=[self.S, self.A])
            self.R = (1-self.epsilon_r)*self.common_reward + self.epsilon_r*self.local_R 
        else:
            self.R = np.random.uniform(0., 1., size=[self.S, self.A]) # reward between [0, 1],  shape [H, S, A]   

    def get_P(self):
        return self.P
    
    def get_r(self):
        return self.R
    
    def sample_specific_triplet(self, state, action):
        p = self.P[state, action]
        r = self.R[state, action]
        next_s = np.random.choice(self.S, 1, p=p)
        next_s_item = next_s.item()
        return next_s_item ,r

    def reset(self,):
        self.state = np.random.randint(self.S)
        return self.state

    def step(self, action):
        r = self.R[self.state, action]
        p = self.P[self.state, action]
        s = np.random.choice(self.S, 1, p=p)
        self.state = s.item()
        return self.state, r
    
    def observation_dim(self):
        return self.S
    
    def action_dim(self):
        return self.A
    
if __name__ == '__main__':
    Env = FiniteStateFiniteActionMDP()

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Challengin_MDP(gym.Env):
    """
    A custom Reinforcement Learning environment with a switching transition kernel
    represented as a NumPy array of shape (S, A, S), with 5 states and 4 actions.

    The transition kernel now directly stores the probability of transitioning
    from state s to state next_s given action a. Rewards are handled separately
    in the step function for simplicity in this representation.
    """
    def __init__(self, k=1):
        super().__init__()
        # Define action and observation space
        self.action_space =[0,1,2,3]  # 4 possible actions
        self.observation_space = [0,1,2,3,4, 5,6]  # 5 possible states (0 to 4)
        self.num_states = 7
        self.num_actions = 4
        self.S = 7
        self.A = 4
        self.k = k

        # Define the two possible transition kernels as NumPy arrays of shape (S, A, S)
        self.transition_kernel_1 = np.array([
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 0
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],  # state 3
            [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.0],  # state 4
            [0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.6, 0.2, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]],

            [[0.0, 0.0, 0.0, 0.1, 0.4, 0.5, 0.0],  # state 5
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]],

            [[0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.2],  # state 6
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]
        ])


        self.transition_kernel_2 = np.array([
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 0
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],  # state 3
            [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.0],  # state 4
            [0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.6, 0.2, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]],

            [[0.0, 0.0, 0.0, 0.1, 0.4, 0.5, 0.0],  # state 5
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]],

            [[0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.2],  # state 6
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]
        ])

        self.reward_function_1 = np.array([
            [0.0, 0.0, 0.0, 0.0],  # state 0
            [1.0, 1.0, 1.0, 1.0],     # state 1
            [1.0, 1.0, 1.0, 1.0],     # state 2
            [0.1, 0.2, 0.0, 0.00],# state 3
            [0.05, 0.05, 0.1, 0.00], # state 4
            [0.1, 0.1, 0.0, 0.00],# state 5: low reward
            [0.1, 0.2, 0.05, 0.00] # state 6: high reward for action 2
        ])


        self.reward_function_2 = np.array([
           [0.0, 0.0, 0.0, 0.0],  # state 0
            [1.0, 1.0, 1.0, 1.0],     # state 1
            [1.0, 1.0, 1.0, 1.0],     # state 2
            [0.1, 0.2, 0.0, 0.00],# state 3
            [0.05, 0.05, 0.1, 0.00], # state 4
            [0.1, 0.1, 0.0, 0.00],# state 5: low reward
            [0.1, 0.2, 0.05, 0.00] # state 6: high reward for action 2
        ])


        # Define the current active transition kernel
        if self.k ==1:
            self.current_transition_kernel = self.transition_kernel_1
            self.reward_function = self.reward_function_1
        else:
            self.current_transition_kernel = self.transition_kernel_2
            self.reward_function = self.reward_function_2

        # Initial state
        self.current_state = 0

    def sample_specific_triplet(self, state, action):
        p = self.current_transition_kernel[state, action]
        r = self.reward_function[state, action]
        next_s = np.random.choice(self.S, 1, p=p)
        next_s_item = next_s.item()
        return next_s_item ,r


    def reset(self):
        weights =[0.05, 0.05,0.05, 0.3, 0.3, 0.2,0.05]
        # Sample one element
        self.current_state = random.choices(self.observation_space , weights=weights, k=1)[0]
        return self.current_state

    def step(self, action):
        # Get the probability distribution over next states for the current state and action
        probabilities = self.current_transition_kernel[self.current_state, action]
        # Sample the next state based on the probabilities
        next_state = np.random.choice(self.num_states, p=probabilities)
        reward = self.reward_function[self.current_state, action]
        self.current_state = next_state
        return next_state, reward

    def get_P(self):
        return self.current_transition_kernel
    
    def get_r(self):
        return self.reward_function
    
    
    
    
class Challengin_Gridword(gym.Env):
    """
    A custom Reinforcement Learning environment with a switching transition kernel
    represented as a NumPy array of shape (S, A, S), with 5 states and 4 actions.

    The transition kernel now directly stores the probability of transitioning
    from state s to state next_s given action a. Rewards are handled separately
    in the step function for simplicity in this representation.
    """
    def __init__(self, k=1):
        super().__init__()
        # Define action and observation space
        self.action_space =[0,1,2,3]  # 4 possible actions
        self.observation_space = [0,1,2,3,4, 5,6]  # 5 possible states (0 to 4)
        self.num_states = 7
        self.num_actions = 4
        self.S = 7
        self.A = 4
        self.k = k

        # Define the two possible transition kernels as NumPy arrays of shape (S, A, S)
        self.transition_kernel_1 = np.array([
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 0
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],  # state 3
            [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.0],  # state 4
            [0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.6, 0.2, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]],

            [[0.0, 0.0, 0.0, 0.1, 0.4, 0.5, 0.0],  # state 5
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]],

            [[0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.2],  # state 6
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]
        ])



        self.transition_kernel_2 = np.array([
            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 0
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # state 1
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # state 2
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],  # state 3
            [0.0, 0.0, 0.0, 0.7, 0.2, 0.1, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0]],

            [[0.0, 0.0, 0.0, 0.6, 0.2, 0.2, 0.0],  # state 4
            [0.0, 0.0, 0.0, 0.5, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.6, 0.2, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0]],

            [[0.0, 0.0, 0.0, 0.1, 0.4, 0.5, 0.0],  # state 5
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.1],
            [0.0, 0.0, 0.0, 0.4, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.8]],

            [[0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.2],  # state 6
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.4],
            [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]]
        ])

        self.reward_function_1 = np.array([
            [0.0, 0.0, 0.0, 0.0],  # state 0
            [0.0, 0.0, 1.0, 0.5],     # state 1
            [0.0, 0.5, 0.0, 0.0],     # state 2
            [0.1, 0.2, 0.0, 0.00],# state 3
            [0.05, 0.05, 0.1, 0.00], # state 4
            [0.1, 0.1, 0.0, 0.00],# state 5: low reward
            [0.1, 0.2, 0.05, 0.00] # state 6: high reward for action 2
        ])


        self.reward_function_2 = np.array([
            [0.0, 0.0, 0.0, 0.0],  # state 0
            [0.0, 0.0, 1.0, 0.5],     # state 1
            [0.0, 0.5, 0.0, 0.0],     # state 2
            [0.1, 0.2, 0.0, 0.00],# state 3
            [0.05, 0.05, 0.1, 0.00], # state 4
            [0.1, 0.1, 0.0, 0.00],# state 5: low reward
            [0.1, 0.2, 0.05, 0.00] # state 6: high reward for action 2
        ])


        # Define the current active transition kernel
        if self.k ==1:
            self.current_transition_kernel = self.transition_kernel_1
            self.reward_function = self.reward_function_1
        else:
            self.current_transition_kernel = self.transition_kernel_2
            self.reward_function = self.reward_function_2

        # Initial state
        self.current_state = 0

    def sample_specific_triplet(self, state, action):
        p = self.current_transition_kernel[state, action]
        r = self.reward_function[state, action]
        next_s = np.random.choice(self.S, 1, p=p)
        next_s_item = next_s.item()
        return next_s_item ,r


    def reset(self):
        weights =[0.05, 0.05,0.05, 0.3, 0.3, 0.2,0.05]
        # Sample one element
        self.current_state = random.choices(self.observation_space , weights=weights, k=1)[0]
        return self.current_state

    def step(self, action):
        # Get the probability distribution over next states for the current state and action
        probabilities = self.current_transition_kernel[self.current_state, action]
        # Sample the next state based on the probabilities
        next_state = np.random.choice(self.num_states, p=probabilities)
        reward = self.reward_function[self.current_state, action]
        self.current_state = next_state
        return next_state, reward

    def get_P(self):
        return self.current_transition_kernel
    
    def get_r(self):
        return self.reward_function