# import packages

import gymnasium as gym
import numpy as np
import pandas as pd
import copy
import math
import random

from MutablePriorityQueue import MutablePriorityQueue
from MassEvacuation import MassEvacuation
from MassEvacuationPolicy import MassEvacuationPolicy


env = MassEvacuation(seed = 20180529)
bm = MassEvacuationPolicy()

# define the default action
action = {'x_hl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
          'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
          'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
         }

numTrials = 30
objectiveValue = np.zeros(numTrials)

for t in range(numTrials):
    done = False

    print('Trial : ', t, ' of', numTrials)
    while not done:

        # The action space in this is defined by three types of actions that can be taken that depend on the event that is occuring.
        # The actions are:
        # - load the helicopter with {white, green, yellow, red}
        # - load the ship with {whtie, green, yellow, red}
        # - unload the ship with {white, green, yellow, red}

        action = {'x_hl_k' : 
                  {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                  'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                  'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
                  }

        if (env.state['e_k']) == 1:

            # load the helicopter
            action['x_hl_k'] = bm.greenFirstLoadingPolicy(env.state, {
                'total_capacity' : env.initial_state['c_h'],
                'individual_capacity' : env.initial_state['delta_h']
            })
        elif (env.state['e_k'] == 2):

            # load the ship
            action['x_sl_k'] = bm.greenFirstLoadingPolicy(env.state, {
                'total_capacity' : env.initial_state['c_s'],
                'individual_capacity' : env.initial_state['delta_s']
            })
        elif (env.state['e_k'] == 3):

            # unload the ship
            action['x_su_k'] = bm.white_unloading_policy(env.state) 

        observation, reward, terminated, truncated, info = env.step(action)
        env.state = observation

        objectiveValue[t] += reward
        done = terminated or truncated

    env.reset(single_scenario = True)

print(objectiveValue)
print('Expected number of lives saved = ', sum(objectiveValue) / numTrials)







