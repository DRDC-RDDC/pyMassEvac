""""
This script demonstrates how the environment can be used to within a learning
algorithm. The class of policy we will elect to learn in this example is a 
Value Function Approximation (VFA), where the approximation strategy will be 
a lookup table representation. In this case we will use a post-decision state 
variable as the approximation; in paticular, we will use a four-dimensional
post-decision state that is the number of individuals in the white, green,
yellow, and red triage categories at the evacuation site. The primary aim 
here is to demonstrate how the environment can be used within a learning 
algorithm, rather than to learn a near-optimal policy.
 
For this demonstration, we will use an Approximate Value Iteration learning
algorithm with a backward pass. The output of this script is two files: 

- first, v_bar.pdf which is a plot of the value of the post-decision state 
after the first decision epoch as a function of iteration;
- second, objective.pdf which is a plot of the expected number of lives saved
and 95% confidence interval as a function of iteration.
"""

import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

import numpy as np
from datetime import datetime
import copy
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from gym_mass_evacuation.mass_evacuation import MassEvacuation

def multichoose(n,k):
    """multichoose is used to compute the set of possible decisions before applying the constraints. 

    This code is taken from the following reference:

    @misc {kuperberg2009a,
        title = {Uniquely generate all permutations of three digits that 
            sum to a particular value?},
        author = {Greg Kuperberg (https://mathoverflow.net/users/1450/
            greg-kuperberg)},
        howpublished = {MathOverflow},
        note = {URL:https://mathoverflow.net/q/9494 (version: 2009-12-21)},
        url = {https://mathoverflow.net/q/9494}
        }

    Parameters
    ----------
    n : integer
        Number of bins.
    k : integer
        Number of items to place in the bins.

    Returns
    -------
    list
        List of all possible ways to place k items in n bins.
    """

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
    """Compute the subset of candidate decisions from all possible decisions 
    given the constraints.

    Parameters
    ----------
    state : dict
        Dictionary that defines the current state S_k.
    initial_state : dict
        Dictionary that defines the initial state S_0.
    x_hl_k : list
        List of possible helicopter loading decisions.
    x_sl_k : list
        List of possible ship loading decisions.
    x_su_k : list
        List of possible ship unloading decisions.

    Returns
    -------
    list
        List of indicies of the valid candidate decisions for the given state.
    """

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

        # Loading a ship
        valid_idx = [idx for idx, x in enumerate(x_sl_k) \
                     if (np.dot(x, delta_s) + \
                         np.sum(rho_s_k) <= initial_state['c_s']) & \
                            (np.all(x <= rho_e_k))]

    if state['e_k'] == 3:

        # Unloading a ship
        valid_idx = [idx for idx, x in enumerate(x_su_k) \
                     if (np.all(x <= rho_s_k))]

    # valid_idx now lists the indices from the corresponding decision that 
    # are valid and are to be considered for either exploration or 
    # exploitation

    return valid_idx


def select_random_decision(state, initial_state, x_hl_k, x_sl_k, x_su_k,
                           x_k_as_dict, valid_idx, v_bar_n_minus_one,
                           t):
    """Select a random decision given the current state and set of candidate
    decisions.

    Parameters
    ----------
    state : dict
        Dictionary that defines the current state S_k.
    initial_state : dict
        Dictionary that defines the initial state S_0.
    x_hl_k : list
        List of possible helicopter loading decisions.
    x_sl_k : list
        List of possible ship loading decisions.
    x_su_k : list
        List of possible ship unloading decisions.
    x_k_as_dict : dict
        Decision x_k as a dict, including x_hl_k, x_sl_k, and x_su_k.
    valid_idx : list
        List of indicies of the valid candidate decisions for the given state.
    v_bar_n_minus_one : list
        List of dictionaries of approximate value functions for each decision
        epoch.
    t : integer
        Current decision epoch.

    Returns
    -------
    dict
        Decision to be taken, including x_hl_k, x_sl_k, and x_su_k.
    ndarray
        Post-decision state S^x_k as a result of the selected decision.
    """

    post_decision_state = np.array([state['rho_e_k']['white'], \
                                    state['rho_e_k']['green'],
                                    state['rho_e_k']['yellow'],
                                    state['rho_e_k']['red']]
                                    )

    # Define the default decision
    decision = None

    if state['e_k'] == 1:

        decision = np.random.choice(valid_idx)

        # compute post-decision state
        post_decision_state = post_decision_state - \
            x_hl_k[decision]

        x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
        x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
        x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
        x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

    if state['e_k'] == 2:
                
        decision = np.random.choice(valid_idx)
                
        # compute post-decision state
        post_decision_state = post_decision_state - \
            x_sl_k[decision]

        x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
        x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
        x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
        x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

    if state['e_k'] == 3:
                
        decision = np.random.choice(valid_idx)

        # compute post-decision state
        post_decision_state = post_decision_state + \
            x_su_k[decision]

        x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
        x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
        x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
        x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

    return x_k_as_dict, post_decision_state


def select_best_decision(state, initial_state, x_hl_k, x_sl_k, x_su_k,
                           x_k_as_dict, valid_idx, v_bar_n_minus_one,
                           t):
    """Select the best decision given the current state, set of candidate
    decisions, and decision policy.

    Parameters
    ----------
    state : dict
        Dictionary that defines the current state S_k.
    initial_state : dict
        Dictionary that defines the initial state S_0.
    x_hl_k : list
        List of possible helicopter loading decisions.
    x_sl_k : list
        List of possible ship loading decisions.
    x_su_k : list
        List of possible ship unloading decisions.
    x_k_as_dict : dict
        Decision x_k as a dict, including x_hl_k, x_sl_k, and x_su_k.
    valid_idx : list
        List of indicies of the valid candidate decisions for the given state.
    v_bar_n_minus_one : list
        List of dictionaries of approximate value functions for each decision
        epoch.
    t : integer
        Current decision epoch.

    Returns
    -------
    dict
        Decision to be taken, including x_hl_k, x_sl_k, and x_su_k.
    ndarray
        Post-decision state S^x_k as a result of the selected decision.
    """

    current_state = np.array([state['rho_e_k']['white'], \
                                    state['rho_e_k']['green'],
                                    state['rho_e_k']['yellow'],
                                    state['rho_e_k']['red']]
    )

    # Set default post-decision state
    post_decision_state = current_state

    # Define the default decision
    decision = None

    # Define the default for the estimate of the value of a decision, v_bar
    v_bar = np.zeros(len(valid_idx))
            
    if (state['e_k'] == 1):
        contribution = [np.sum(x_hl_k[i]) for i in valid_idx]

        post_decision_state = [current_state - \
            x_hl_k[i] for i in valid_idx]

        keys = [str(x) for x in post_decision_state]
        for i, key in enumerate(keys):
            if key in v_bar_n_minus_one[t]:
                v_bar[i] = contribution[i] + v_bar_n_minus_one[t][key]
            else:
                v_bar[i] = contribution[i]

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        if len(idx) > 0:
            decision = valid_idx[np.random.choice(idx)]

    if (state['e_k'] == 2):
        contribution = [np.sum(x_sl_k[i]) for i in valid_idx]

        post_decision_state = [current_state - \
            x_sl_k[i] for i in valid_idx]

        keys = [str(x) for x in post_decision_state]

        for i, key in enumerate(keys):
            if key in v_bar_n_minus_one[t]:
                v_bar[i] = contribution[i] + v_bar_n_minus_one[t][key]
            else:
                v_bar[i] = contribution[i]

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        if len(idx) > 0:
            decision = valid_idx[np.random.choice(idx)]

    if (state['e_k'] == 3):

        post_decision_state = [current_state + \
            x_su_k[i] for i in valid_idx]

        keys = [str(x) for x in post_decision_state]
        for i, key in enumerate(keys):
            if key in v_bar_n_minus_one[t]:
                v_bar[i] = v_bar_n_minus_one[t][key]
            else:
                v_bar[i] = 0

        idx = np.argwhere(v_bar == np.max(v_bar)).flatten().tolist()

        if len(idx) > 0:
            decision = valid_idx[np.random.choice(idx)]

    if decision is not None:
        if (state['e_k'] == 1):

            x_k_as_dict['x_hl_k']['white'] = x_hl_k[decision][0]
            x_k_as_dict['x_hl_k']['green'] = x_hl_k[decision][1]
            x_k_as_dict['x_hl_k']['yellow'] = x_hl_k[decision][2]
            x_k_as_dict['x_hl_k']['red'] = x_hl_k[decision][3]

            post_decision_state = current_state - \
                        x_hl_k[decision]

        if (state['e_k'] == 2):

            x_k_as_dict['x_sl_k']['white'] = x_sl_k[decision][0]
            x_k_as_dict['x_sl_k']['green'] = x_sl_k[decision][1]
            x_k_as_dict['x_sl_k']['yellow'] = x_sl_k[decision][2]
            x_k_as_dict['x_sl_k']['red'] = x_sl_k[decision][3]

            post_decision_state = current_state - \
                x_sl_k[decision]

        if (state['e_k'] == 3):

            x_k_as_dict['x_su_k']['white'] = x_su_k[decision][0]
            x_k_as_dict['x_su_k']['green'] = x_su_k[decision][1]
            x_k_as_dict['x_su_k']['yellow'] = x_su_k[decision][2]
            x_k_as_dict['x_su_k']['red'] = x_su_k[decision][3]

            post_decision_state = current_state + \
                    x_su_k[decision]

    return x_k_as_dict, post_decision_state


def sideSimulation(initial_state, x_hl_k, x_sl_k, x_su_k, v_bar_n, \
    single_scenario, seed):
    """Side simulation of the mass evacuation environment to be executed 
    after a specified number of iterations of the Approximate Value
    Iteration algorithm.

    Parameters
    ----------
    initial_state : dict
        Dictionary that defines the initial state S_0.
    x_hl_k : list
        List of possible helicopter loading decisions.
    x_sl_k : list
        List of possible ship loading decisions.
    x_su_k : list
        List of possible ship unloading decisions.
    v_bar_n : list
        List of dictionaries of approximate value functions for each decision
        epoch.
    single_scenario : boolean
        True, reset the environment to use the initial individual transition times; False, sample new initial transition times.
    seed : integer
        Seed to be used when creating the mass evacuation environment.

    Returns
    -------
    dict
        Consists of one key, 'objective', that is the objective that resulted
        by following the VFA decision policy with \bar{V} set to v_bar_n.
    """

    objective = 0
    
    # Create an environment
    env = MassEvacuation(initial_state = initial_state, \
                         seed = seed, \
                            default_rng = True)

    # Set the initial state
    S_k, info = env.reset(options = {'single_scenario' : single_scenario})

    side_done = False
    l = 0

    while (not side_done):
    # while (not side_done) and (env.state['tau_k'] <= 400):
        
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

        x_k_as_dict, _ = \
                    select_best_decision(env.state, initial_state, 
                                         x_hl_k, x_sl_k, x_su_k, 
                                         x_k_as_dict, 
                                         valid_idx, 
                                         v_bar_n,
                                         l)

        if (l == 1) & (seed == 624327):
            print(x_k_as_dict)

        # given the selected decision, get the next state
        x_k = env.action_dict_to_ndarray(x_k_as_dict)

        S_k_plus_one, reward, terminated, truncated, info = env.step(x_k)

        S_k_plus_one_as_dict = env.observation_ndarray_to_dict(S_k_plus_one)
        
        env.state = S_k_plus_one_as_dict
        
        objective += reward
        l += 1
                
        side_done = terminated or truncated   

    if seed == 624327:
        print('Objective is', objective)

    return {'objective': objective}

def avi_agent(initial_state, seed, max_iterations, sim_at_iteration, \
    num_sim_at_iteration, eta, single_scenario):
    """Approximate Value Iteration learning algorithm.

    Parameters
    ----------
    initial_state : dict
        Dictionary that defines the initial state S_0.
    seed : integer
        Random number generator seed.
    max_iterations : integer
        Number of iterations to run the approximate value iteration algorithm.
    sim_at_iteration : integer
        Number of iterations between running the side simulation using v_bar_n.
    eta : float
        Between zero and one. Defines the percentage of decisions that are taken
        randomly during learning.
    single_scenario : boolean
        True, reset the environment to use the initial individual transition times; False, sample new initial transition times.        

    Returns
    -------
    dict
        Consists of two keys: 'v_bar_n' that is a list of post-decision states and their values for each decision epoch; and 'expected_objective', that a pandas data frame that has three columns, 'eta', 'Iteration', and
        'Value' where each row is the value of eta used, the iteration number,
        and value of the post-decision state at iteration zero.
    """

    N = max_iterations
    M = sim_at_iteration
    O = num_sim_at_iteration

    # create the gymnasium environment
    env = MassEvacuation(initial_state = initial_state, \
        seed = seed, default_rng = True)

    max_at_evac = \
        np.sum(np.array(list(initial_state['rho_e_k'].values())))

    # maximum number of decision epochs that will be considered
    max_episode_time = 400
    max_epochs = 400

    v_bar_n = []
    v_bar_n_minus_one = []

    for _ in range(max_epochs):
        v_bar_n.append({})

    v_bar_n_minus_one = copy.deepcopy(v_bar_n)

    # Create a random number generator for the exploration / exploitation
    rng = np.random.default_rng(seed = seed)

    # Compute all possible decisions to load a helicopter (x^{hl}_k)
    x_hl_k = list()
    for i in range(initial_state['c_h'] + 1):
    
        x = multichoose(4, i)
        for j in range(len(x)):
            x_hl_k.append(x[j])

    # Compute all possible decisions to load a a ship (x^{sl}_k)
    x_sl_k = list()
    for i in range(initial_state['c_s'] + 1):
    
        x = multichoose(4, i)
        for j in range(len(x)):
            x_sl_k.append(x[j])

    # Compute all possible decisions to load a a ship (x^{su}_k)
    x_su_k = list()
    for i in range(initial_state['c_s'] + 1):
    
        x = multichoose(4, i)
        for j in range(len(x)):
            x_su_k.append(x[j])

    # Define the learning rate alpha_n
    alpha_n = 0.05

    # Define the two data frames - v_bar and expected_objective - that are
    # used to return the results
    v_bar = pd.DataFrame(columns = ['eta', 'Iteration', 'Value'])
    expected_objective = pd.DataFrame(columns = ['eta', 'Iteration', \
        'Objective'])

    # Set the side simulation counter
    m = 0

    # Set of seeds that will be used to start the side simulations
    side_sim_seeds = np.random.default_rng(seed = 4837294)
    seeds = side_sim_seeds.integers(low = 1, high = 1e6, size = O)

    for n in range(N):

        print('Starting iteration', n)

        # Set the initial state
        S_k, info = env.reset(options = {'single_scenario' : single_scenario})

        # Reset the post-decision state
        post_decision_state_list = list()

        # episode loop counter
        k = 0
        
        done = False

        while (not done) and (env.state['tau_k'] <= max_episode_time):
            
            # As described in Rempel (2024), p. 5, x_k is a twelve-dimensional 
            # vector, x_k = (x^hl_k, x^sl_k, x^su_k)
            x_k = np.zeros(12, dtype = np.int64)

            # convert the ndarray to a dictionary so that we can more easily specify the decision
            x_k_as_dict = env.action_ndarray_to_dict(x_k)

            valid_idx = apply_constraints(env.state, initial_state,
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

                x_k_as_dict, post_decision_state = \
                    select_random_decision(env.state, initial_state, 
                                    x_hl_k, x_sl_k, x_su_k, 
                                    x_k_as_dict, 
                                    valid_idx,
                                    v_bar_n_minus_one,
                                    k)
                
            else:

                # Exploitation - solve the optimization problem for the 
                # VFA-based decision policy to find the best $x_k$

                x_k_as_dict, post_decision_state = \
                select_best_decision(env.state, initial_state, 
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
                
            key = str(post_decision_state_list[t - 1]['pds'])

            if key in v_bar_n_minus_one[t - 1]:
                v_bar_n[t - 1][key] = \
                        (1 - alpha_n) * v_bar_n_minus_one[t - 1][key] + \
                        alpha_n * v_hat_t
            else:
                v_bar_n[t - 1][key] = alpha_n * v_hat_t

        v_bar_n_minus_one = copy.deepcopy(v_bar_n)

        # Run a side simulation to determine if the algorithm has leanred a 
        # better expected objective value. Do this every M iterations.

        if (n % M == 0):

            # create a pool for multiprocessing
            pool = mp.Pool(mp.cpu_count())

            # Run a side simulation to evaluate how the learning is proceeding
            objective = [pool.apply_async(sideSimulation, 
                                    args=(rempel_2024_initial_state, x_hl_k, x_sl_k, x_su_k, v_bar_n, single_scenario, seed)) for seed in seeds]

            pool.close()
            pool.join()

            for o in range(O):
                tmp = pd.Series(data = {'eta' : eta, 'Iteration' : n, \
                    'Objective' : objective[o].get()['objective']})
                
                expected_objective = pd.concat([expected_objective, \
                    tmp.to_frame().T], ignore_index = True)

            tmp = pd.Series(data = {'eta' : eta, 'Iteration' : n, \
                'Value' : np.max(np.array(list(v_bar_n[1].values())))})
            
            v_bar = pd.concat([v_bar, tmp.to_frame().T], ignore_index = True)

            m += 1

    return {'v_bar' : v_bar, 'expected_objective' : expected_objective}

if __name__ == '__main__':

    print('Starting Approximate Value Iteration algorithm ...')

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
        'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 25, 'red' : 5},
        'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
        'initial_helo_arrival' : [0],
        'initial_ship_arrival' : [1000]
        }

    # Learn a decision policy
    single_scenario = True
    num_iterations = 100
    iterations_between_side_sim = 10
    num_side_simulations = 30


    print('Running algorithm with eta = 1, full exploration')

    eta = 1
    eta_1 = avi_agent(rempel_2024_initial_state, rempel_2024_seed, \
        num_iterations, iterations_between_side_sim, \
            num_side_simulations, eta, single_scenario)

    print('Running algorithm with eta = 0.25, balanced exploration and exploitation')

    eta = 0.25
    eta_25 = avi_agent(rempel_2024_initial_state, rempel_2024_seed, \
        num_iterations, iterations_between_side_sim, \
            num_side_simulations, eta, single_scenario)

    # Create a plot of the value of the post-decision state at iteration 0
    # as a function of iteration.

    v_bar = pd.concat([eta_1['v_bar'], eta_25['v_bar']], ignore_index = True)
    sns.relplot(data = v_bar, x = 'Iteration', y = 'Value', hue = 'eta', \
        kind = 'line', markers = True, style = 'eta', palette = 'Set2')
    plt.savefig('v_bar.pdf')
    plt.close()

    # Create a plot of the expected objective (number of individuals saved)
    # and the 95% confidence interval based on the side simualtion, as a 
    # function of iteration.

    objective = pd.concat([eta_1['expected_objective'], \
        eta_25['expected_objective']], ignore_index = True)
    
    fig, ax = plt.subplots()
    
    sns.lineplot(data = objective, \
        x = 'Iteration', y = 'Objective', hue = 'eta', markers = True, \
            style = 'eta', palette = 'Set2')
    
    ax.set_ylim(0, np.sum(np.array(list(rempel_2024_initial_state['rho_e_k'].values()))))
    ax.set_ylabel('Expected number of lives saved')

    plt.savefig('objective.pdf')
    plt.close()

    print('Done.')