# -*- coding: utf-8 -*-
"""
Provides Memory class

@author: Jan Bayer
"""

import cv2
import numpy as np
from collections import deque
import tensorflow_probability as tfp
tfd = tfp.distributions

DATA_FOLDER = None

class Memory():
    ''' Memory class for the replay memory
    '''
    def __init__(self, memory_size: int, expert: bool):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.expert = expert

    def add(self, experience):
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        indexes = np.random.choice(np.arange(len(self.buffer)), 
                                   size=batch_size, replace=False)
        if self.expert:
            output = []
            for idx in indexes:
                # Get imgs
                state = cv2.imread(DATA_FOLDER + self.buffer[idx][0],
                                   cv2.IMREAD_UNCHANGED).reshape([84,84,4])
                next_state = cv2.imread(DATA_FOLDER + self.buffer[idx][1],
                                        cv2.IMREAD_UNCHANGED).reshape([84,84,4])
                output.append((state, next_state, self.buffer[idx][2],
                               self.buffer[idx][3], self.buffer[idx][4]))
            return output
        else:
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)
    
    def load(self, obs, actions, rewards, episode_starts):
        ''' Loads expert with imgs - not optimal, very slow
            and memory demanding
        '''
        # Get state and next_state from the imgs
        # next_state list is moved one obs from state list
        state = []
        for i, ob in enumerate(obs[0:self.memory_size+1]):
            print(ob)
            img = cv2.imread(DATA_FOLDER + obs[i], cv2.IMREAD_UNCHANGED)
            state.append(img.reshape([84, 84, 4]))
        
        # Create done array from episode_starts
        done = [0] * obs.shape[0]
        episode_starts[0] = False
        idxs = np.where(episode_starts == True)[0] - 1
        for idx in idxs:
            done[idx] = 1
        done[-1] = 1
        
        for i in range(self.memory_size): 
            self.add((state[i], state[i+1], actions[i].squeeze(), 1, done[i]))
            
    def load_expert(self, obs, actions, rewards, episode_starts):
        ''' Loads expert without imgs - imgs are obtained in samples
        ''' 
        # Create done array from episode_starts
        done = [0] * obs.shape[0]
        episode_starts[0] = False
        idxs = np.where(episode_starts == True)[0] - 1
        for idx in idxs:
            done[idx] = 1
        done[-1] = 1
        
        for i in range(self.memory_size-1): 
            self.add((obs[i], obs[i+1], actions[i].squeeze(), 1, done[i]))

