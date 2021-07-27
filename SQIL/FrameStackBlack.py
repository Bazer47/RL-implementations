# -*- coding: utf-8 -*-
"""
Provides modification of FrameStack

@author: Jan Bayer
"""

import numpy as np
import gym

class FrameStackBlack(gym.wrappers.FrameStack):
    ''' Modifies FrameStack wrapper to init FrameStack with black screens
        as it is in the expert recorded_images
    '''
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(np.zeros_like(observation))
         for _ in range(self.num_stack-1)]
        self.frames.append(observation)
        return self._get_observation()  
