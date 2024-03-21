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


# Set the seed that was used in Rempel (2024)
rempel_2024_seed = 20180529

# Set the initial state
rempel_2024_initial_state = {
            'm_e' : {'white' : 120, 'green' : 48, 'yellow' : 8, 'red' : 1.5},
            'm_s' : {'green' : 48, 'yellow' : 72, 'red' : 120},
            'c_h' : 10,
            'c_s' : 50,
            'delta_h' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'delta_s' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'eta_h' : 3,
            'eta_sl' : 24,
            'eta_su' : 1,
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0},
            'initial_helo_arrival' : [48],
            'initial_ship_arrival' : [0]
        }

env = MassEvacuation(initial_state = rempel_2024_initial_state, \
                     seed = rempel_2024_seed,
                     default_rng = False)

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







