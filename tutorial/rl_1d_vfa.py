#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
import copy

from gym_mass_evacuation.mass_evacuation import MassEvacuation
from gym_mass_evacuation.mass_evacuation_policy import MassEvacuationPolicy

import multiprocessing as mp


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

    if state['e_k'] == 2:

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
    
    # Define the return values
    v_hat_n = 0
    post_decision_state = 0

    # Set the default post-decision state to the what it would be if no decision
    # was taken and no exogneous information arrived
    post_decision_state = state['rho_e_k']['red']

    if state['e_k'] == 1:

        decision = rng.choice(valid_idx)

        # compute v_hat_n
        contribution = np.sum(x_hl_k[decision])

        # compute post-decision state
        post_decision_state = state['rho_e_k']['red'] - \
            x_hl_k[decision][3]

        v_hat_n = contribution + \
            v_bar_n_minus_one[t, post_decision_state]

        x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
        x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
        x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
        x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

    if state['e_k'] == 2:
                
        decision = rng.choice(valid_idx)

        # compute v_hat_n
        contribution = 0
                
        # compute post-decision state
        post_decision_state = state['rho_e_k']['red'] - \
            x_sl_k[decision][3]

        v_hat_n = contribution + \
            v_bar_n_minus_one[t, post_decision_state]

        x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
        x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
        x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
        x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

    if state['e_k'] == 3:
                
        decision = rng.choice(valid_idx)

        # compute v_hat_n
        contribution = 0

        # compute post-decision state
        post_decision_state = state['rho_e_k']['red'] + \
            x_su_k[decision][3]

        v_hat_n = contribution + \
            v_bar_n_minus_one[t, post_decision_state]

        x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
        x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
        x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
        x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

    return x_k_as_dict, post_decision_state


# In[11]:

def select_best_decision(state, initial_state, x_hl_k, x_sl_k, x_su_k,
                           x_k_as_dict, valid_idx, v_bar_n_minus_one,
                           t):

    v_hat_n = 0
    post_decision_state = 0

    # Set the default post-decision state to the what it would be if no decision
    # was taken and no exogneous information arrived
    post_decision_state = state['rho_e_k']['red']
            
    if (state['e_k'] == 1):
        contribution = [np.sum(x_hl_k[i]) for i in valid_idx]

        post_decision_state = [state['rho_e_k']['red'] - \
            x_hl_k[i][3] for i in valid_idx]

        v_bar = contribution + v_bar_n_minus_one[t, post_decision_state]

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        decision = valid_idx[np.random.choice(idx)]

    if (state['e_k'] == 2):
        contribution = 0

        post_decision_state = [state['rho_e_k']['red'] - \
            x_sl_k[i][3] for i in valid_idx]

        v_bar = contribution + v_bar_n_minus_one[t, post_decision_state]

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        decision = valid_idx[np.random.choice(idx)]

    if (state['e_k'] == 3):
        contribution = 0

        post_decision_state = [state['rho_e_k']['red'] + \
            x_su_k[i][3] for i in valid_idx]

        v_bar = contribution + v_bar_n_minus_one[t, post_decision_state]

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        decision = valid_idx[np.random.choice(idx)]

    if (state['e_k'] == 1):

        x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
        x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
        x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
        x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

        post_decision_state = state['rho_e_k']['red'] - \
                x_hl_k[decision][3]

    if (state['e_k'] == 2):

        x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
        x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
        x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
        x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

        post_decision_state = state['rho_e_k']['red'] - \
                x_sl_k[decision][3]   

    if (state['e_k'] == 3):

        x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
        x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
        x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
        x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

        post_decision_state = state['rho_e_k']['red'] + \
                x_su_k[decision][3]

    return x_k_as_dict, post_decision_state


def sideSimulation(initial_state, x_hl_k, x_sl_k, x_su_k, v_bar_n):

    objective = 0
    
    # Create an environment
    env = MassEvacuation(initial_state = initial_state, 
                         seed = None, default_rng = True)

    # Set the initial state
    S_k, info = env.reset(options = {'single_scenario' : True})

    side_done = False
    l = 0

    while (not side_done) and (env.state['tau_k'] <= 400):
        
        # As described in Rempel (2024), p. 5, x_k is a 
        # twelve-dimensional vector, x_k = (x^hl_k, x^sl_k, x^su_k)
        x_k = np.zeros(12, dtype = np.int64)

        # convert the ndarray to a dictionary so that we can more 
        # easily specify the decision
        x_k_as_dict = env.action_ndarray_to_dict(x_k)

        valid_idx = apply_constraints(env.state, initial_state, 
                                              x_hl_k, 
                                              x_sl_k, 
                                              x_su_k)

        x_k_as_dict, post_decision_state = \
                    select_best_decision(env.state, initial_state, 
                                         x_hl_k, x_sl_k, x_su_k, 
                                         x_k_as_dict, 
                                         valid_idx, 
                                         v_bar_n,
                                         l)

        # given the selected decision, get the next state
        x_k = env.action_dict_to_ndarray(x_k_as_dict)

        S_k_plus_one, reward, terminated, truncated, info = env.step(x_k)

        S_k_plus_one_as_dict = env.observation_ndarray_to_dict(S_k_plus_one)
        
        env.state = S_k_plus_one_as_dict
        
        objective += reward
        l += 1
                
        side_done = terminated or truncated   

    return {'objective': objective}


# In[20]:


# Approximate Value Iteration (AVI) learning algorithm

if __name__ == '__main__':

    # set the environment's seed and initial state that was used in Rempel 
    # (2024)
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
        'initial_helo_arrival' : [0],
        'initial_ship_arrival' : [1000]
        }

    # create the gymnasium environment
    env = MassEvacuation(initial_state = rempel_2024_initial_state, \
        seed = rempel_2024_seed, default_rng = True)

    max_at_evac = np.sum(np.array(list(rempel_2024_initial_state['rho_e_k'].values())))
    # max_on_ship = rempel_2024_initial_state['c_s']

    # maximum number of decision epochs that will be considered
    T = 400

    v_bar_n = np.full(shape = (T, max_at_evac + 1), \
        fill_value = 0, dtype = np.float64)
    v_bar_n_minus_one = np.full(shape = (T, max_at_evac + 1), \
        fill_value = 0, dtype = np.float64)

    # Set the maximum number of iterations
    N = 1000

    # Set the exploration probability
    eta = 1

    # Set the number of iterations that should pass before a side simulation is 
    # run
    M = 10

    # Set the learning rate - using a generalized harmonic stepsize, alpha = 
    # a / (a + n - 1) see Powell (2011), p. 430
    a = 25

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

    alpha_n = 0.05

    print('Using multiprocessing with', mp.cpu_count(), 'cores')

    expected_objective = np.zeros(N)
    m = 0

    for n in range(N):

        print('Starting iteration', n)

        # Set the initial state
        S_k, info = env.reset(options = {'single_scenario' : True})

        # Reset the post-decision state
        post_decision_state_list = list()

        # loop counter
        k = 0
        
        done = False

        while (not done) and (env.state['tau_k'] <= 400):
            
            # As described in Rempel (2024), p. 5, x_k is a twelve-dimensional 
            # vector, x_k = (x^hl_k, x^sl_k, x^su_k)
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
                                    k)
                
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
                                    k)

            # Update the list of post-decision states
            post_decision_state_list.append({'x_hl_k' : np.array(list(x_k_as_dict['x_hl_k'].values())),
                                            'pds' : post_decision_state})
            
            # given the selected decision, get the next state
            
            x_k = env.action_dict_to_ndarray(x_k_as_dict)

            S_k_plus_one, reward, terminated, truncated, info = env.step(x_k)

            S_k_plus_one_as_dict = env.observation_ndarray_to_dict(S_k_plus_one)
            env.state = S_k_plus_one_as_dict

            k += 1
            
            done = terminated or truncated    

        # Update the values of the post-decision states using a backward pass
        v_hat_t = 0
        for t in range(k - 1, 0, -1):

            # compute the backward pass value for v^hat
            if t == k - 1:
                v_hat_t = np.sum(post_decision_state_list[t]['x_hl_k'])
            else:
                v_hat_t = np.sum(post_decision_state_list[t]['x_hl_k']) + v_hat_t
                
            v_bar_n[t - 1, post_decision_state_list[t - 1]['pds']] = \
                    (1 - alpha_n) * v_bar_n_minus_one[t - 1, post_decision_state_list[t - 1]['pds']] + \
                    alpha_n * v_hat_t

        v_bar_n_minus_one = np.copy(v_bar_n)

        #if (n > 0):
        #    alpha_n = a / (a + n - 1)

        # Run a side simulation to determine if the algorithm has leanred a better 
        # expected objective value. Do this every M iterations.

        if (n % M == 0):

            # create a pool for multiprocessing
            pool = mp.Pool(mp.cpu_count())

            # Run a side simulation to evaluate how the learning is proceeding
            objective = [pool.apply_async(sideSimulation, 
                                    args=(rempel_2024_initial_state, x_hl_k, x_sl_k, x_su_k, v_bar_n)) for _ in range(30)]

            pool.close()
            pool.join()

            for z in range(30):
                expected_objective[m] += objective[z].get()['objective']
            expected_objective = expected_objective / 30

            print('After', n, 'iterations, expected objective value is', \
                expected_objective[m])

            m += 1
