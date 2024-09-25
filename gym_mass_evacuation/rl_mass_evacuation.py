# import packages

# import gymnasium as gym
import numpy as np
import pandas as pd
import copy
import math
import random

# import local copies for testing purposes
from mass_evacuation import MassEvacuation
from mass_evacuation_policy import MassEvacuationPolicy

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
            'eta_sl' : 1,
            'eta_su' : 24,
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0},
            'initial_helo_arrival' : [48],
            'initial_ship_arrival' : [168]
        }

env = MassEvacuation(initial_state = rempel_2024_initial_state, \
                     seed = rempel_2024_seed,
                     default_rng = False)

bm = MassEvacuationPolicy(default_rng = False)

# define the default action
action = {'x_hl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
          'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
          'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
         }

num_trials = 30
objective_value = np.zeros(num_trials)

for t in range(num_trials):
    done = False

    # system_time = 0

    print('Trial : ', t, ' of', num_trials)
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

        # # FOR TESTING ONLY - state before action is taken
        # print('Sysem time -->', system_time)
        # print('Current event queue -->')
        # print(env.queue.queue)
        # print('State -->', env.state)

        if (env.state['e_k']) == 1:

            # load the helicopter
            action['x_hl_k'] = bm.green_first_loading_policy(env.state, {
                'total_capacity' : env.initial_state['c_h'],
                'individual_capacity' : env.initial_state['delta_h']
            })
        elif (env.state['e_k'] == 2):

            # load the ship
            action['x_sl_k'] = bm.green_first_loading_policy(env.state, {
                'total_capacity' : env.initial_state['c_s'],
                'individual_capacity' : env.initial_state['delta_s']
            })
        elif (env.state['e_k'] == 3):

            # unload the ship
            action['x_su_k'] = bm.white_unloading_policy(env.state) 

        observation, reward, terminated, truncated, info = env.step(action)
        env.state = observation

        # system_time += env.state['tau_k']

        objective_value[t] += reward
        done = terminated or truncated

    options = {}
    options['single_scenario'] = True
    env.reset(options)

print(objective_value)
print('Expected number of lives saved = ', objective_value.mean())

##### ----- testing lerning algorithms -----

env = MassEvacuation(initial_state = rempel_2024_initial_state, \
                     seed = rempel_2024_seed,
                     default_rng = False)

print(env.state)

# sample an action
action_sample = env.action_space.sample()

# convert the sampled action into an action that can be used by the environment
action = {'x_hl_k' : {'white' : action_sample['x_hl_k'][0,0], 
                      'green' : action_sample['x_hl_k'][0,1], 
                      'yellow' : action_sample['x_hl_k'][0,2], 
                      'red' : action_sample['x_hl_k'][0,3]}, \
          'x_sl_k' : {'white' : action_sample['x_sl_k'][0,0], 
                      'green' : action_sample['x_sl_k'][0,1], 
                      'yellow' : action_sample['x_sl_k'][0,2],
                      'red' : action_sample['x_sl_k'][0,3]},
          'x_su_k' : {'white' : action_sample['x_su_k'][0,0], 
                      'green' : action_sample['x_su_k'][0,1], 
                      'yellow' : action_sample['x_su_k'][0,2], 
                      'red' : action_sample['x_su_k'][0,3]}
                  }

print(action)

# submit the action to the environment and check what is returned
observation, reward, terminated, truncated, info = env.step(action)
print(observation)
