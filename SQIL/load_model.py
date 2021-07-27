# -*- coding: utf-8 -*-
"""
Run trained NN model
Needs to load the weights from ckpt file

@author: Jan Bayer
"""

import gym
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
import pickle
import tensorflow_probability as tfp
tfd = tfp.distributions
from NeuralNetwork import NeuralNetwork
from FrameStackBlack import FrameStackBlack

plt.close('all')

# Define path to the ckpt file
ckpt_path = 'checkpoints/SQIL.ckpt' 

# Define path to the log file of the training history
log_path = 'logs/SQIL_pickle'

# Data folder with the expert
DATA_FOLDER = 'data/'

# If the rendered images should be stored
STORE_IMGS = False

# %% Matplotlib setup

def mm2inch(*tupl):
    inch = 25.4
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.size' : 8,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'axes.titlesize': 8,
    'lines.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.markeredgewidth': 0.8,
    'lines.markersize': 6,
    'lines.markerfacecolor': 'none',
    'savefig.dpi': 200
})

# %% Expert demonstrations processing

def calc_reward_per_episode(episode_starts, rewards, n_of_episodes):
    rewards_per_episode = np.zeros(n_of_episodes, dtype=np.int32)
    counter = 0
    ''' Calculates reward per episode from the expert data
    '''
    
    for i, start in enumerate(episode_starts):

        if start == True and i > 0:
            counter += 1
        else:
            rewards_per_episode[counter] += rewards[i]
        
    return rewards_per_episode

def timesteps_per_episode(episode_starts):
    ''' Creates array with timesteps per episode 
    '''
    true_values = np.where(episode_starts == True)
    true_values = np.append(true_values, (len(episode_starts)-1))
    return np.diff(true_values).squeeze()

# Load expert data
actions_exp = np.load(DATA_FOLDER + 'actions.npy')
episode_starts_exp = np.load(DATA_FOLDER + 'episode_starts.npy')
obs_exp = np.load(DATA_FOLDER + 'obs.npy')
rewards_exp= np.load(DATA_FOLDER + 'rewards.npy')

# Create array with the lenght of number of episodes
episodes_exp = np.arange(1,np.count_nonzero(episode_starts_exp)+1)

# Calc rewards per episode as only rewards per timestep were provided 
rewards_per_episode_exp = calc_reward_per_episode(episode_starts_exp, rewards_exp,
                                              len(episodes_exp))

timesteps_exp = timesteps_per_episode(episode_starts_exp)

mean_exp = np.mean(rewards_per_episode_exp) 
min_exp = min(rewards_per_episode_exp)
max_exp = max(rewards_per_episode_exp)

# %% Neural network

# Create policy network and load trained weights
policy_network = NeuralNetwork()
policy_network.create_soft_q_network()
policy_network.model.load_weights(ckpt_path)

# %% Environment

env = gym.make('BreakoutNoFrameskip-v0').unwrapped
env = gym.wrappers.AtariPreprocessing(env)
env = FrameStackBlack(env, 4)
env.seed(42)
state = env.reset()

# %% Playing


# Init history dict
all_data = {'actions':[], 'rewards':[], 'done':[], 'episode':[],
            'episode_reward' : []}

for episode in range(2):
    state = env.reset()
    episode_reward = 0
    print(episode)
    for t in range(1000):
        # env.render()
        if STORE_IMGS == True:
            img = env.render(mode='rgb_array')
            cv2.imwrite(f'imgs/img_{episode}_{t}.png', img)
        action = policy_network.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        
        # Store history
        all_data['actions'].append(int(action.numpy()))
        all_data['rewards'].append(reward)
        all_data['done'].append(done)
        all_data['episode'].append(episode)
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        
    all_data['episode_reward'].append(episode_reward)
    
env.close()

# %% Postprocessing of the model

mean_model = np.mean(all_data['episode_reward']) 
min_model = min(all_data['episode_reward'])
max_model = max(all_data['episode_reward'])
 
# Load training data
f = open(log_path,'rb')
all_data_learning = pickle.load(f)

all_data_learning['loss_np'] = [all_data_learning['loss'][i].numpy()
                                for i in range(len(all_data_learning['loss']))]
# %% Plots

# Comparison between the expert and model
plt.figure(figsize=mm2inch(150,60))
plt.plot(episodes_exp, rewards_per_episode_exp, label = 'expert')
plt.plot(episodes_exp, all_data['episode_reward'], label = 'model')
plt.xlabel('episode')
plt.xlim((0,1000))
plt.ylabel('reward')
plt.legend()
plt.tight_layout(pad=0.1)

# Training loss
plt.figure(figsize=mm2inch(150,60))
plt.plot(range(800, len(all_data_learning['loss'])+800),
         all_data_learning['loss_np'])
plt.xlabel('timestep')
plt.xlim((800,len(all_data_learning['loss'])+800))
plt.ylabel('loss')
# plt.legend()
plt.tight_layout(pad=0.1)

# Reward per episode
plt.figure(figsize=mm2inch(150,60))
plt.plot(range(0, len(all_data_learning['episode_reward'])),
         all_data_learning['episode_reward'])
plt.xlabel('episode')
plt.ylabel('reward')
plt.xlim((0,200))
# plt.legend()
plt.tight_layout(pad=0.1)