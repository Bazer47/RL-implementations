# -*- coding: utf-8 -*-
"""
Main training script

Implementation of SQIL: Imitation Learning via Reinforcement
Learning with Sparse Rewards (Reddy et al. (2019))

@author: Jan Bayer
"""

import numpy as np
import gym
import os
import pickle
import datetime
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
from NeuralNetwork import NeuralNetwork
from FrameStackBlack import FrameStackBlack
from Memory import Memory

# %% Setup

# Data folder with expert npy and recorded_images
DATA_FOLDER = 'data/'

time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Save TensorFlow model weights - path
checkpoint_path = 'checkpoints/SQIL_' + time + '.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

# Save all data dict - path
pickle_path = 'logs/SQIL_pickle_' + time

# Seed for environment
seed = 42

# %% Expert load
# You need to import expert's playing!

actions = np.load(DATA_FOLDER + 'actions.npy')
episode_starts = np.load(DATA_FOLDER + 'episode_starts.npy')
obs = np.load(DATA_FOLDER + 'obs.npy')
rewards = np.load(DATA_FOLDER + 'rewards.npy')

# %% Environment

env = gym.make('BreakoutNoFrameskip-v0').unwrapped
env = gym.wrappers.AtariPreprocessing(env)
env = FrameStackBlack(env, 4)
env.seed(seed)
state = env.reset()

# %% Neural network setup

# Create policy and target network 
policy_network = NeuralNetwork()
policy_network.create_soft_q_network()

target_network = NeuralNetwork()
target_network.create_soft_q_network()

target_network.model.set_weights(policy_network.model.get_weights())

# Set optimizer and loss for neural network training
optimizer = keras.optimizers.Adam(learning_rate=0.00025)
loss_function = tf.keras.losses.MeanSquaredError()
# %% Training

GAMMA = 0.99 # Discount parameter
REPLAY_MEMORY = actions.shape[0]*2
BATCH = 32
UPDATE_STEPS = 1000 # When the target network is updated

# Init history dict
all_data = {'actions':[], 'rewards':[], 'done':[], 'episode':[],
            'episode_reward' : [], 'loss' : []}

# Replay memory is divided into expert experience and online experience (1:1)
expert_replay_memory = Memory(REPLAY_MEMORY//2, expert=True)
expert_replay_memory.load_expert(obs, actions, rewards, episode_starts)
online_replay_memory = Memory(REPLAY_MEMORY//2, expert=False)

learn_steps = 0
begin_learn = False
episode_reward = 0

for episode in range(800):
        state = env.reset()
        episode_reward = 0
        print('Episode: ', episode)
        for time_steps in range(1000):
            # env.render()
            action = policy_network.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # Store online experience (state, next_state, act, reward, done)
            online_replay_memory.add((np.array(state).reshape([84,84,4]),
                                      np.array(next_state).reshape([84,84,4]),
                                      tf.squeeze(action),
                                      0,
                                      int(done)))
            
            # Store history
            all_data['actions'].append(int(action.numpy()))
            all_data['rewards'].append(reward)
            all_data['done'].append(done)
            all_data['episode'].append(episode)
            
            # Training section
            if online_replay_memory.size() > 800:
                # Wait until there is some online experience
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                learn_steps += 1
                print('Learning step: ', learn_steps)
                
                if learn_steps % UPDATE_STEPS == 0:
                    # Update target_network
                    target_network.model.set_weights(policy_network.model.get_weights())
                
                # Get online batch
                online_batch = online_replay_memory.sample(BATCH//2)
                online_batch_state, online_batch_next_state,\
                    online_batch_action, online_batch_reward,\
                        online_batch_done = zip(*online_batch)

                online_batch_state = tf.convert_to_tensor(online_batch_state)
                online_batch_next_state = tf.convert_to_tensor(online_batch_next_state)
                online_batch_action = tf.convert_to_tensor(online_batch_action)
                online_batch_reward = tf.convert_to_tensor(online_batch_reward)
                online_batch_done = tf.convert_to_tensor(online_batch_done)
                
                # Get expert batch
                expert_batch = expert_replay_memory.sample(BATCH//2)
                expert_batch_state, expert_batch_next_state,\
                    expert_batch_action, expert_batch_reward,\
                        expert_batch_done = zip(*expert_batch)

                expert_batch_state = tf.convert_to_tensor(expert_batch_state)
                expert_batch_next_state = tf.convert_to_tensor(expert_batch_next_state)
                expert_batch_action = tf.convert_to_tensor(expert_batch_action)
                expert_batch_reward = tf.convert_to_tensor(expert_batch_reward)
                expert_batch_done = tf.convert_to_tensor(expert_batch_done)
                
                # Concate expert and online batch (1:1)
                batch_state = tf.concat([online_batch_state, 
                                         expert_batch_state], 0)
                batch_next_state = tf.concat([online_batch_next_state,
                                              expert_batch_next_state], 0)
                batch_action = tf.concat([online_batch_action,
                                          expert_batch_action], 0)
                batch_reward = tf.concat([online_batch_reward,
                                          expert_batch_reward], 0)
                batch_done = tf.concat([online_batch_done,
                                        expert_batch_done], 0)
                
                # Calc q_hat soft via the soft Bellman eq.
                next_q = target_network.model(batch_next_state)
                next_v = target_network.getVsoft(next_q)
                q_hat = (tf.cast(batch_reward, tf.float32) 
                         + (1 - tf.cast(batch_done, tf.float32)) 
                         * tf.constant(GAMMA) 
                         * tf.squeeze(next_v))
                
                # Create mask to have only q_values with the action sample
                masks = tf.one_hot(batch_action, 4)

                with tf.GradientTape() as tape:
                    
                    # Actual q_values
                    q_values = policy_network.model(batch_state)
                    
                    # Squared soft Bellman error
                    loss = loss_function(tf.reduce_sum(
                        tf.multiply(q_values, masks), axis=1),
                        q_hat)
                    print('Loss: ', loss)
                    all_data['loss'].append(loss)
                
                # Backpropagation
                grads = tape.gradient(loss,
                                      policy_network.model.trainable_variables)
                optimizer.apply_gradients(zip(grads,
                                              policy_network.model.trainable_variables))
            
            if done:
                break
            
            # Update state
            state = next_state
            
        print('Episode reward', episode_reward)
        all_data['episode_reward'].append(episode_reward)
        
        if episode % 10 == 0:
            
            # Store model weights
            policy_network.model.save_weights(checkpoint_path)
            # Pickle all data
            f = open(pickle_path,'wb')
            pickle.dump(all_data,f)
            
env.close()