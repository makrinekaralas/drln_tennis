import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from DDPG_agent import Agent

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
UPDATE_EVERY = 10        # how often to update the network
GAMMA = 0.99            # discount factor


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaDDPGAgent():
    """Multi-agent interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, warmup, random_seed, nb_agents):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            nb_agents (int): number of agents
            warmup (int): number of iterations to warm-up before using policy
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.nb_agents = nb_agents
        self.warmup = warmup
        
        self.agents = [Agent(state_size=self.state_size, action_size=self.action_size, warmup=self.warmup, random_seed=i) for i in range(self.nb_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def reset(self):
        """Reset every agent noise."""
        for agent in self.agents:
            agent.reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        """Add experience to shared ReplayBuffer and take step for each agent"""
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                for agent in self.agents:
                    experiences = self.memory.sample()
                    agent.learn(experiences, GAMMA)

    def act(self, states, timestep, add_noise=True):
        """Act for each agent"""
        return [agent.act(np.expand_dims(state, axis=0), timestep) for agent, state in zip(self.agents, states)]


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)