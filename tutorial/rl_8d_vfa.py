#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import gymnasium as gym
import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env

from gym_mass_evacuation.mass_evacuation import MassEvacuation
from gym_mass_evacuation.mass_evacuation_policy import MassEvacuationPolicy


# In[2]:


# set the environment's seed and initial state that was used in Rempel (2024)
rempel_2024_seed = 20180529

# set the initial state
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
                'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5},
                'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'initial_helo_arrival' : [48],
                'initial_ship_arrival' : [0]
}


# In[3]:


# create the gymnasium environment
env = MassEvacuation(initial_state = rempel_2024_initial_state, 
                         seed = rempel_2024_seed, default_rng = True)

# check that the environment is valid
check_env(env)


# We will now demonstrate how the environment can be used to learn a decision policy. The class of policy we will elect to learn in this example is a Value Function Approximation (VFA), where the approximation strategy will be a lookup table representation similar to Q-learning. However, rather than using the full state variable $S_k$, in this case we will use a post-decision state variable. In paticular, we will use a simple post-decision state as the primary aim here is to demonstrate how the environment can be used with a learning algorithm, rather than to learn a near-optimal policy.
# 
# Our VFA policy is given as
# 
# $X^{VFA}(S_k) = \underset{x_k \in \mathcal{X_k}}{\arg \text{max}}\; C(S_k, x_k) + \bar{V}(S^x_k)$,
# 
# where $\bar{V}(S^x_k)$ represents the value of being in the post-decision state $S^x_k = S^{M,x}(S_k, x_k)$. 
# 
# For our demonstration, we will use a two-dimensional post-decision state similar to that used in Rempel et al. (2021). However, in this case the post-decision state will be given as
# 
# $S^x_k = [\sum_{t \in \mathcal{T}} \rho^e_{k,t+1}, \sum_{t \in \mathcal{T}} \rho^s_{k,t+1}]$,
# 
# where the notation follows that found in Rempel (2024).

# In[5]:


# multichoose is used to compute the set of possible decisions before applying the constraints. 

# this code is taken from the following reference
# @MISC {9494,
#   TITLE = {Uniquely generate all permutations of three digits that sum to a particular value?},
#    AUTHOR = {Greg Kuperberg (https://mathoverflow.net/users/1450/greg-kuperberg)},
#    HOWPUBLISHED = {MathOverflow},
#    NOTE = {URL:https://mathoverflow.net/q/9494 (version: 2009-12-21)},
#    EPRINT = {https://mathoverflow.net/q/9494},
#    URL = {https://mathoverflow.net/q/9494}
#}

def multichoose(n,k):
    
    if k < 0 or n < 0: 
        return "Error"
    
    if not k: 
        return [[0]*n]
    
    if not n: 
        return []
    
    if n == 1: 
        return [[k]]
    
    return [[0]+val for val in multichoose(n-1,k)] + \
        [[val[0]+1]+val[1:] for val in multichoose(n,k-1)]


# In[6]:


# Given the initial number of individuals at the evacuation site, i.e., 2000, there are 2000 post-decision states. We will store this as a two-dimensional numpy array.

max_at_evac = np.sum(np.array(list(rempel_2024_initial_state['rho_e_k'].values())))
# max_on_ship = rempel_2024_initial_state['c_s']

# maximum number of decision epochs that will be considered
T = 400

v_bar_n = []
v_bar_n_minus_one = []
for i in range(T):
    v_bar_n.append({})

v_bar_n_minus_one = copy.deepcopy(v_bar_n)

# Set the maximum number of iterations
N = 1000

# Set the exploration probability
eta = 0.25

# Set the learning rate - using a generalized harmonic stepsize, alpha = a / (a + n - 1) 
# see Powell (2011), p. 430
a = 10

# Set the iteration counter
n = 1

# Create a random number generator for the exploration / exploitation
rng = np.random.default_rng(seed = rempel_2024_seed)

# Define a variable to capture the k - 1 post-decision state
# post_decision_state = {'evac_site' : 0}
# post_decision_state_k_minus_one = {'evac_site' : 0}

# Compute all possible decisions to load a helicopter (x^{hl}_k)
x_hl_k = list()
for i in range(rempel_2024_initial_state['c_h'] + 1):
    
    x = multichoose(4, i)
    for j in range(len(x)):
        x_hl_k.append(x[j])

# Compute all possible decisions to load a a ship (x^{sl}_k)
x_sl_k = list()
for i in range(rempel_2024_initial_state['c_s'] + 1):
    
    x = multichoose(4, i)
    for j in range(len(x)):
        x_sl_k.append(x[j])

# Compute all possible decisions to load a a ship (x^{su}_k)
x_su_k = list()
for i in range(rempel_2024_initial_state['c_s'] + 1):
    
    x = multichoose(4, i)
    for j in range(len(x)):
        x_su_k.append(x[j])

# Before considering the set of constraints $\mathcal{X}_k$, there are the following number of candidate 
# decisions given the initial state $S_0$:

# In[7]:


print('Loading a helicopter: ', len(x_hl_k))


# In[8]:


# Define a function to find all decisions that are valid given the current 
# state $S_k$ and the complete set of all possible decisions.

def apply_constraints(state, initial_state, x_hl_k, x_sl_k, x_su_k):

    # idx is a list of the indicies of the possible decisions that are valid 
    # given the current state $S_k$ and constraints $\mathcal{X}_k$. These 
    # indicies are from:
    #
    # if e_k == 1: x_hl_k
    # if e_k == 2: x_sl_k
    # if e_k == 3: x_su_k
        
    valid_idx = []

    delta_h = np.array(list(initial_state['delta_h'].values()))
    delta_s = np.array(list(initial_state['delta_s'].values()))
    rho_e_k = np.array(list(state['rho_e_k'].values()))
    rho_s_k = np.array(list(state['rho_s_k'].values()))
    
    # Given the current state, filter the set of all possible decisions to the 
    # valid set of decisions
    if state['e_k'] == 1:

        # Loading a helicopter
        valid_idx = [idx for idx, x in enumerate(x_hl_k) 
                     if (np.dot(x, delta_h) <= initial_state['c_h']) & \
                    (np.all(x <= rho_e_k))]

    if env.state['e_k'] == 2:

        valid_idx = [idx for idx, x in enumerate(x_sl_k) \
                     if (np.dot(x, delta_s) + \
                         np.sum(rho_s_k) <= initial_state['c_s']) & \
                            (np.all(x <= rho_e_k))]

    if state['e_k'] == 3:

        valid_idx = [idx for idx, x in enumerate(x_su_k) \
                     if (np.all(x <= rho_s_k))]

    # valid_idx now lists the indices from the corresponding decision that 
    # are valid and are to be considered for either exploration or 
    # exploitation

    return valid_idx 


# In[9]:


def select_random_decision(state, initial_state, x_hl_k, x_sl_k, x_su_k,
                           x_k_as_dict, valid_idx, v_bar_n_minus_one,
                           t):

    v_hat_n = 0

    # Set the default post-decision state to the what it would be if no decision
    # was taken and no exogneous information arrived
    post_decision_state = np.array([state['rho_e_k']['white'], \
                                    state['rho_e_k']['green'],
                                    state['rho_e_k']['yellow'],
                                    state['rho_e_k']['red'],
                                    state['rho_s_k']['white'],
                                    state['rho_s_k']['green'],
                                    state['rho_s_k']['yellow'],
                                    state['rho_s_k']['red']])

    if state['e_k'] == 1:

        decision = rng.choice(valid_idx)

        # compute the contribution
        contribution = np.sum(x_hl_k[decision])

        # compute post-decision state
        post_decision_state = post_decision_state - \
            np.concatenate([x_hl_k[decision], np.zeros(4)])

        key = str(post_decision_state)
        if key in v_bar_n_minus_one[t]:
            v_hat_n = contribution + v_bar_n_minus_one[t][key]
        else:
            v_hat_n = contribution

        x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
        x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
        x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
        x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

    if state['e_k'] == 2:
                
        decision = rng.choice(valid_idx)

        # compute the contribution
        contribution = 0
                
        # compute post-decision state
        post_decision_state = post_decision_state - \
            np.concatenate([x_sl_k[decision], np.zeros(4)]) + \
                np.concatenate([np.zeros(4), x_sl_k[decision]])

        key = str(post_decision_state)
        if key in v_bar_n_minus_one[t]:
            v_hat_n = contribution + v_bar_n_minus_one[t][key]
        else:
            v_hat_n = contribution

        x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
        x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
        x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
        x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

    if state['e_k'] == 3:
                
        decision = rng.choice(valid_idx)

        # compute the contribution
        contribution = 0
                
        # compute post-decision state
        post_decision_state = post_decision_state + \
            np.concatenate([x_su_k[decision], np.zeros(4)]) - \
                np.concatenate([np.zeros(4), x_su_k[decision]])

        key = str(post_decision_state)
        if key in v_bar_n_minus_one[t]:
            v_hat_n = contribution + v_bar_n_minus_one[t][key]
        else:
            v_hat_n = contribution

        x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
        x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
        x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
        x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

    return x_k_as_dict, post_decision_state


# In[11]:


def select_best_decision(state, initial_state, x_hl_k, x_sl_k, x_su_k,
                           x_k_as_dict, valid_idx, v_bar_n_minus_one,
                           t):

    v_hat = 0

    # Set the default post-decision state to the what it would be if no decision
    # was taken and no exogneous information arrived
    current_state = np.array([state['rho_e_k']['white'], \
                                    state['rho_e_k']['green'],
                                    state['rho_e_k']['yellow'],
                                    state['rho_e_k']['red'],
                                    state['rho_s_k']['white'],
                                    state['rho_s_k']['green'],
                                    state['rho_s_k']['yellow'],
                                    state['rho_s_k']['red']])
            
    candidate_decisions = pd.DataFrame(columns = ['index', 
                                                  'contribution', 'v_bar', 'v_hat'], 
                                                  dtype = pd.Float64Dtype())

    for i in valid_idx:

        if (state['e_k'] == 1):

            # compute the contribution
            contribution = np.sum(x_hl_k[i])

            # compute post-decision state
            post_decision_state = current_state - \
                np.concatenate([x_hl_k[i], np.zeros(4)])

            key = str(post_decision_state)
            if key in v_bar_n_minus_one[t]:
                v_bar = v_bar_n_minus_one[t][key]
                v_hat = contribution + v_bar
            else:
                v_bar = 0
                v_hat = contribution
     
            new_row = pd.Series({'index' : i, \
                'contribution' : contribution, 
                'v_bar' : v_bar, 
                'v_hat' : v_hat})
                    
            candidate_decisions = pd.concat([candidate_decisions, 
                                                     new_row.to_frame().T], ignore_index = True)

        if (state['e_k'] == 2):

            # compute the contribution
            contribution = 0
                
            # compute post-decision state
            post_decision_state = current_state - \
                np.concatenate([x_sl_k[i], np.zeros(4)]) + \
                    np.concatenate([np.zeros(4), x_sl_k[i]])

            key = str(post_decision_state)
            if key in v_bar_n_minus_one[t]:
                v_bar = v_bar_n_minus_one[t][key]
                v_hat = contribution + v_bar
            else:
                v_bar = 0
                v_hat = contribution

            new_row = pd.Series({'index' : i, 
                                         'contribution' : contribution, 
                                         'v_bar' : v_bar,
                                         'v_hat' : v_hat}) 
            
            candidate_decisions = pd.concat([candidate_decisions, \
                                                     new_row.to_frame().T], 
                                                     ignore_index = True)

        if (state['e_k'] == 3):

            # compute the contribution
            contribution = 0
                
            # compute post-decision state
            post_decision_state = current_state + \
                np.concatenate([x_su_k[i], np.zeros(4)]) - \
                    np.concatenate([np.zeros(4), x_su_k[i]])

            key = str(post_decision_state)
            if key in v_bar_n_minus_one[t]:
                v_bar = v_bar_n_minus_one[t][key]
                v_hat = contribution + v_bar
            else:
                v_bar = 0
                v_hat = contribution

            new_row = pd.Series({'index' : i, 
                                         'contribution' : contribution, 
                                         'v_bar' : v_bar, 
                                         'v_hat' : v_hat}
                                         )
            
            candidate_decisions = pd.concat([candidate_decisions, 
                                                     new_row.to_frame().T], ignore_index = True)

    # search the pandas dataframe to determine which decision provides 
    # the maximum value (v_hat)

    if (len(valid_idx) > 0):
        best_decision = \
            candidate_decisions.query('v_hat == v_hat.max()')
        decision = int(best_decision['index'].sample())
    else:
        decision = 0

    if (state['e_k'] == 0):

        post_decision_state = current_state

    if (state['e_k'] == 1):

        x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
        x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
        x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
        x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

        post_decision_state = current_state - \
            np.concatenate([x_hl_k[decision], np.zeros(4)])

    if (state['e_k'] == 2):

        x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
        x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
        x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
        x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

        post_decision_state = current_state - \
            np.concatenate([x_sl_k[decision], np.zeros(4)]) + \
                np.concatenate([np.zeros(4), x_sl_k[decision]])

    if (state['e_k'] == 3):

        x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
        x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
        x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
        x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

        post_decision_state = current_state + \
            np.concatenate([x_su_k[decision], np.zeros(4)]) - \
                np.concatenate([np.zeros(4), x_su_k[decision]])

    return x_k_as_dict, post_decision_state


# In[20]:


# Approximate Value Iteration (AVI) learning algorithm

objective = np.zeros(N)
alpha_n = 0.95
average = 0

for n in range(N):

    # Set the initial state
    S_k, info = env.reset(options = {'single_scenario' : True})

    # Reset the post-decision state
    post_decision_state_list = list()

    # loop counter
    t = 0
    
    done = False

    while (not done) and (env.state['tau_k'] <= 400):
        
        # As described in Rempel (2024), p. 5, x_k is a twelve-dimensional vector, 
        # x_k = (x^hl_k, x^sl_k, x^su_k)
        x_k = np.zeros(12, dtype = np.int64)

        # convert the ndarray to a dictionary so that we can more easily specify the decision
        x_k_as_dict = env.action_ndarray_to_dict(x_k)

        valid_idx = apply_constraints(env.state, rempel_2024_initial_state,
                                      x_hl_k,
                                      x_sl_k,
                                      x_su_k)

        # generate a random number from a uniform distribution that will be 
        # used to determine if this decision will focus on exploring or 
        # exploitating
        r = rng.random()

        if r <= eta:

            # Exploration - randomly select a decision from 
            # $x_k \in \mathcal{X}_k$

            x_k_as_dict, post_decision_state = select_random_decision(env.state, rempel_2024_initial_state, 
                                   x_hl_k, x_sl_k, x_su_k, 
                                   x_k_as_dict, 
                                   valid_idx,
                                   v_bar_n_minus_one,
                                   t)
            
        else:

            # Exploitation - solve the optimization problem for the VFA-based 
            # decision policy to find the best $x_k$

            x_k_as_dict, post_decision_state = \
            select_best_decision(env.state, 
                                   rempel_2024_initial_state, 
                                   x_hl_k, x_sl_k, x_su_k, 
                                   x_k_as_dict, 
                                   valid_idx,
                                   v_bar_n_minus_one,
                                   t)

        # Update the list of post-decision states
        post_decision_state_list.append({'x_hl_k' : np.array(list(x_k_as_dict['x_hl_k'].values())),
                                        'pds' : post_decision_state})
        
        # given the selected decision, get the next state
        
        x_k = env.action_dict_to_ndarray(x_k_as_dict)

        S_k_plus_one, reward, terminated, truncated, info = env.step(x_k)

        S_k_plus_one_as_dict = env.observation_ndarray_to_dict(S_k_plus_one)
        env.state = S_k_plus_one_as_dict
        
        objective[n] += reward
        t += 1
        
        done = terminated or truncated    

    # Update the values of the post-decision states using a backward pass
    v_hat_t = 0
    for l in range(t - 1, 0, -1):

        # compute the backward pass value for v^hat
        if l == t - 1:
            v_hat_t = np.sum(post_decision_state_list[l]['x_hl_k'])
        else:
            v_hat_t = np.sum(post_decision_state_list[l]['x_hl_k']) + \
                v_hat_t
            
        key = str(post_decision_state_list[l - 1]['pds'])
        if key in v_bar_n_minus_one[l - 1]:

            v_bar_n[l - 1][key] = \
                (1 - alpha_n) * v_bar_n_minus_one[l - 1][key] + alpha_n * v_hat_t
        else:

            v_bar_n[l - 1][key] = alpha_n * v_hat_t

    v_bar_n_minus_one = copy.deepcopy(v_bar_n)

    if (n > 0):
        alpha_n = a / (a + n - 1)
        average = (1 - 0.05) * average + 0.05 * objective[n]
    else:
        average = objective[n]

    print('Iteration ', n,' complete --- objective =', objective[n], 'with smoothed average of', average)

    # if (n % 25 == 0):
    #     fig, ax = plt.subplots()
    #     im = ax.imshow(v_bar_n)
    #     ax.set_xlabel('Number of people onboard the ship')
    #     ax.set_ylabel('Number of people at the evac site')

    #     cbar = ax.figure.colorbar(im, ax=ax)
    #     cbar.ax.set_ylabel('Value function', rotation=-90, va="bottom")
    #     fig.tight_layout()

    #     plt.show()
    


# In[ ]:


import seaborn as sns

sns.relplot(x = np.arange(0, len(objective), 1), y = objective, kind = 'line')

