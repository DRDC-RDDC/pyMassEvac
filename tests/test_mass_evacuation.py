import pytest
import numpy as np
import pandas as pd
import copy

from gym_mass_evacuation import mass_evacuation

@pytest.fixture
def initial_state():

    return {
            'm_e' : {'white' : 120, 'green' : 48, 'yellow' : 8, 'red' : 1.5},
            'm_s' : {'green' : 48, 'yellow' : 72, 'red' : 120},
            'c_h' : 40,
            'c_s' : 100,
            'delta_h' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'delta_s' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'eta_h' : 3,
            'eta_sl' : 48,
            'eta_su' : 1,
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0},
            'initial_helo_arrival' : [48],
            'initial_ship_arrival' : [16]
        }

@pytest.fixture
def seed():

    return 20180529

def test_init(initial_state, seed):
    """Test the init method.

    Test the init method.

    Parameters
    ----------
    initial_state : dict
        Initial state as defined in the pytest.fixture.
    seed : int
        Seed as defined in the pytest.fixture.    
    """

    # Create the mass evacuation environment
    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the expected state
    expected_state = {'tau_k' : 0,
                      'e_k' : 0,
                      'rho_e_k' : initial_state['rho_e_k'],
                      'rho_s_k' : initial_state['rho_s_k']}

    # Set the expected queue
    expected_queue = pd.DataFrame(columns = ['tau_k', 'e_k'])
    ship = pd.DataFrame({'tau_k': initial_state['initial_ship_arrival'][0], \
                         'e_k' : 2}, index = [0])
    helo = pd.DataFrame({'tau_k': initial_state['initial_helo_arrival'][0] - \
                         initial_state['initial_ship_arrival'][0], \
                         'e_k' : 1}, index = [0])
    expected_queue = pd.concat([expected_queue, ship, helo], \
                               ignore_index = True)

    # Set the expected shape of the exogenous information data frames
    expected_exog_evac_shape = (sum(initial_state['rho_e_k'].values()), 7)
    expected_exog_ship_shape = (0, 7)

    assert env.initial_state == initial_state
    assert env.state == expected_state
    assert env.queue.queue.equals(expected_queue)
    assert env.exog_med_transitions_evac.shape == expected_exog_evac_shape
    assert env.exog_med_transitions_ship.shape == expected_exog_ship_shape


def test_render(initial_state, seed):
    """Test the render method.

    Test the render method. Currently the method performs no actions.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    expected_result = None

    assert expected_result == env.render()

def test_close(initial_state, seed):
    """Test the close method.

    Test the close method. Currently the method performs no actions.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    expected_result = None

    assert expected_result == env.close()

def test_observation(initial_state, seed):
    """Test the observation method.

    Test the observation method. This method returns the current state S_k.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    env.state = {
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0}
    }

    expected_result = env.state

    assert expected_result == env.observation()

def test_reset_1(initial_state, seed):
    """Test the reset method when single_scenario = True.

    Test the reset method when single_scenario = True. In this situation, the
    initial start state of the environment is that which was set when the
    environment was initially created.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    expected_initial_state = copy.deepcopy(env.initial_state)
    expected_state = copy.deepcopy(env.state)
    expected_queue = copy.deepcopy(env.queue)
    expected_exog_med_transitions_evac = copy.deepcopy( \
        env.exog_med_transitions_evac)
    expected_exog_med_transitions_ship = copy.deepcopy( \
        env.exog_med_transitions_ship)

    env.reset(single_scenario = True)

    assert expected_initial_state == env.initial_state
    assert expected_state == env.state
    assert expected_queue.queue.equals(env.queue.queue)
    assert expected_exog_med_transitions_evac.equals( \
        env.exog_med_transitions_evac)
    assert expected_exog_med_transitions_ship.equals( \
        env.exog_med_transitions_ship)
    
def test_reset_2(initial_state, seed):
    """Test the reset method when single_scenario = False.

    Test the reset method when single_scenario = False. In this situation, the
    initial start state of the environment is that which was set when the
    environment was initially created.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    expected_initial_state = copy.deepcopy(env.initial_state)
    expected_state = copy.deepcopy(env.state)
    expected_queue = copy.deepcopy(env.queue)
    expected_exog_med_transitions_evac = copy.deepcopy( \
        env.exog_med_transitions_evac)
    expected_exog_med_transitions_ship = copy.deepcopy( \
        env.exog_med_transitions_ship)

    env.reset(single_scenario = False)

    assert expected_initial_state == env.initial_state
    assert expected_state == env.state
    assert expected_queue.queue.equals(env.queue.queue)
    assert expected_exog_med_transitions_evac.equals( \
        env.exog_med_transitions_evac) == False
    assert expected_exog_med_transitions_ship.equals( \
        env.exog_med_transitions_ship)

def test_compute_reward_1(initial_state, seed):
    """Test the _compute_reward function when e_k = 1.

    Test the _compute_reward function when the state variable e_k is 1. This
    event is a helicopter loading event, and thus a decision should result in
    an immediate contribution being received. 
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # set the state variable e_k to helicopter loading
    env.state['e_k'] = 1

    # set the helicopter loading decision
    decision = {'x_hl_k' : {'white' : 10, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
    }

    # define the expected reward
    expected_reward = 10

    assert expected_reward == env._compute_reward(decision)

def test_compute_reward_2(initial_state, seed):
    """Test the _compute_reward function when e_k = 2.

    Test the _compute_reward function when the state variable e_k is 2. This
    event is a ship loading event, and thus a decision should result in
    an immediate contribution being received. 
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # set the state variable e_k to helicopter loading
    env.state['e_k'] = 2

    # set the helicopter loading decision
    decision = {'x_hl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_sl_k' : {'white' : 10, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
    }

    # define the expected reward - should be zero since the current event is
    # to load the ship
    expected_reward = 0

    assert expected_reward == env._compute_reward(decision)

def test_add_individuals_1(initial_state, seed):
    """Test the _add_individuals() method when individuals are to be added
    to the ship.

    Test the _add_individuals() method when individuals are to be added to 
    the ship. Note that _add_individuals() adds rows to the exog_med_transition_
    ship data frame, and does not change the state variable. The method does
    not check if space is available onboard the ship---it assumes that this 
    check has already been performed before this method is called.
    """
    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 3, 'e_k' : 2, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 0}
    }

    # Set the state variable in the environment
    env.state = S_k

    # Set the location for the individuals to be loaded
    location = 'ship'

    # Set the decision. Note that at present this method takes single dict
    # as the decision, not a dict with the three decisions for x_hl_k,
    # x_sl_k, and x_su_k. Thus, this decision reflects x_sl_k.
    decision = {'white' : 0, 'green' : 5, 'yellow' : 5, 'red' : 0, 'black' : 0}

    # Compute the expected return value - the addition of rows to the 
    # exogenous medical transition data frame
    expected_result = env.exog_med_transitions_ship.shape[0] + \
        sum(decision.values())
    
    env._add_individuals(decision, location)

    assert expected_result == env.exog_med_transitions_ship.shape[0]
    
def test_add_individuals_2(initial_state, seed):
    """Test the _add_individuals() method when individuals are to be added
    to the evacuation site.

    Test the _add_individuals() method when individuals are to be added to 
    the evacuation site. Note that _add_individuals() adds rows to the 
    exog_med_transition_evac data frame, and does not change the state variable. 
    """
    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 24, 'e_k' : 3, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 0}
    }

    # Set the state variable in the environment
    env.state = S_k

    # Set the location for the individuals to be loaded
    location = 'evac'

    # Set the decision. Note that at present this method takes single dict
    # as the decision, not a dict with the three decisions for x_hl_k,
    # x_sl_k, and x_su_k. Thus, this decision reflects x_sl_k.
    decision = {'white' : 10, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0}

    # Compute the expected return value - the addition of rows to the 
    # exogenous medical transition data frame
    expected_result = env.exog_med_transitions_evac.shape[0] + \
        sum(decision.values())
    
    env._add_individuals(decision, location)

    assert expected_result == env.exog_med_transitions_evac.shape[0]

def test_remove_individuals_1(initial_state, seed):
    """Test the _remove_individuals() method when individuals are to be removed
    from the evacuation site.

    Test the _add_individuals() method when individuals are to be removed from 
    the evacuation site. Note that _remove_individuals() removes rows from the 
    exog_med_transition_evac data frame, and does not change the state variable. 
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the location for the individuals to be loaded
    location = 'evac'

    # Set the decision. Note that at present this method takes single dict
    # as the decision, not a dict with the three decisions for x_hl_k,
    # x_sl_k, and x_su_k. Thus, this decision reflects x_sl_k.
    decision = {'white' : 0, 'green' : 5, 'yellow' : 5, 'red' : 0, 'black' : 0}

    # Compute the expected return value - the addition of rows to the 
    # exogenous medical transition data frame
    expected_result = env.exog_med_transitions_evac.shape[0] - \
        sum(decision.values())
    
    env._remove_individuals(decision, location)

    assert expected_result == env.exog_med_transitions_evac.shape[0]

def test_remove_individuals_2(initial_state, seed):
    """Test the _remove_individuals() method when individuals are to be removed
    from the ship.

    Test the _add_individuals() method when individuals are to be removed from 
    the ship. Note that _remove_individuals() removes rows from the 
    exog_med_transition_ship data frame, and does not change the state variable. 
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the location for the individuals to be loaded
    location = 'ship'

    # First, we need to add individuals to the ship as the initial state
    # of env provides a scenario where all individuals are at the evacuation
    # site. 
    env._add_individuals({'white' : 20, 'green' : 10, 'yellow' : 5, \
                           'red' : 5, 'black' : 0}, 'ship')

    # Set the decision. Note that at present this method takes single dict
    # as the decision, not a dict with the three decisions for x_hl_k,
    # x_sl_k, and x_su_k. Thus, this decision reflects x_sl_k.
    decision = {'white' : 0, 'green' : 5, 'yellow' : 5, 'red' : 0, 'black' : 0}

    # Compute the expected return value - the addition of rows to the 
    # exogenous medical transition data frame
    expected_result = env.exog_med_transitions_ship.shape[0] - \
        sum(decision.values())
    
    env._remove_individuals(decision, location)

    assert expected_result == env.exog_med_transitions_ship.shape[0]

def test_update_medical_condition_1(initial_state, seed):
    """Test the _update_medical_condition method when the location is the 
    evacuation site.

    Test the _update_medical_condition() method for individuals at the 
    evacuation site. Note that method only computes delta_hat_e_k or 
    delta_hat_s_k, based on the value of the location parameter that is
    passed to the method.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    tau_hat_k = 16
    location = 'evac'

    # Remove the rows from the exog_med_transitions data frame and add
    # specific rows for the test case
    env.exog_med_transitions_evac.drop(env.exog_med_transitions_evac.index, inplace = True)

    individual = {}
    individual['arrival_time'] = 0
    individual['category'] = 'white'
    individual['white'] = 12
    individual['green'] = 15
    individual['yellow'] = 18
    individual['red'] = 21
    individual['black'] = np.nan

    env.exog_med_transitions_evac = pd.concat([env.exog_med_transitions_evac, \
                                              pd.DataFrame(individual, index = [0])],
                                              ignore_index = True)
    
    expected_result = {'white' : 0, 'green' : 0, 'yellow' : 1, 'red' : 0, \
                       'black' : 0}
    
    delta_hat_e_k = env._update_medical_condition(tau_hat_k, location)

    assert expected_result == delta_hat_e_k

def test_update_medical_condition_2(initial_state, seed):
    """Test the _update_medical_condition method when the location is the 
    ship.

    Test the _update_medical_condition() method for individuals onboard 
    the ship. Note that method only computes delta_hat_e_k or 
    delta_hat_s_k, based on the value of the location parameter that is
    passed to the method.    
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    tau_hat_k = 48
    location = 'ship'

    # Remove the rows from the exog_med_transitions data frame and add
    # specific rows for the test case
    env.exog_med_transitions_ship.drop(env.exog_med_transitions_ship.index, inplace = True)

    individual = {}
    individual['arrival_time'] = 0
    individual['category'] = 'red'
    individual['white'] = np.nan
    individual['green'] = 48
    individual['yellow'] = 24
    individual['red'] = 12
    individual['black'] = np.nan

    env.exog_med_transitions_ship = pd.concat([env.exog_med_transitions_ship, \
                                              pd.DataFrame(individual, index = [0])],
                                              ignore_index = True)
    
    expected_result = {'white' : 1, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                       'black' : 0}
    
    delta_hat_s_k = env._update_medical_condition(tau_hat_k, location)

    assert expected_result == delta_hat_s_k

def test_compute_delta_hat_k_1(initial_state, seed):
    """Test the _compute_delta_hat_k method when the event is that all
    individuals have arrived at the evacuation site.

    Test the _compute_delta_hat() method when the event is that all
    individuals have arrived at the evacuation site (e_k == 0). Note that 
    method returns both delta_hat_e_k and delta_hat_s_k, and calls several
    of the environment's other methods based on the state variable e_k.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set tau_hat_k to 12 hours and the decision to null, i.e., no individual 
    # moves.
    tau_hat_k = 12
    decision = {'x_hl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
    }

    env.state['e_k'] = 0

    # Remove the rows from the exog_med_transitions data frame and add
    # specific rows for the test case
    env.exog_med_transitions_evac.drop(env.exog_med_transitions_evac.index, inplace = True)

    individual = {}
    individual['arrival_time'] = 0
    individual['category'] = 'white'
    individual['white'] = 12
    individual['green'] = 15
    individual['yellow'] = 18
    individual['red'] = 21
    individual['black'] = np.nan

    env.exog_med_transitions_evac = pd.concat([env.exog_med_transitions_evac, \
                                              pd.DataFrame(individual, index = [0])],
                                              ignore_index = True)

    expected_result = {'delta_hat_e_k' : {'white' : 0, 'green' : 1, \
                                          'yellow' : 0, 'red' : 0, \
                                            'black' : 0}, \
                        'delta_hat_s_k' : {'white' : 0, 'green' : 0, \
                                           'yellow' : 0, 'red' : 0, \
                                            'black' : 0}
    }

    result = env._compute_delta_hat_k(tau_hat_k, decision)

    assert expected_result == result

def test_compute_delta_hat_k_2(initial_state, seed):
    """Test the _compute_delta_hat_k method when the event is that individuals
    are to be loaded onto the helicopter.

    Test the _compute_delta_hat() method when the event is that 
    individuals are to be loaded onto the helicopter (e_k == 1). Note that 
    method returns both delta_hat_e_k and delta_hat_s_k, and calls several
    of the environment's other methods based on the state variable e_k.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set tau_hat_k to 12 hours and the decision to null, i.e., no individual 
    # moves.
    tau_hat_k = 12
    decision = {'x_hl_k' : {'white' : 0, 'green' : 10, 'yellow' : 0, 'red' : 0},
                'x_sl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
    }

    env.state['e_k'] = 1

    # Remove the rows from the exog_med_transitions data frame and add
    # specific rows for the test case
    env.exog_med_transitions_evac.drop(env.exog_med_transitions_evac.index, inplace = True)

    for _ in range(12):
        individual = {}
        individual['arrival_time'] = 0
        individual['category'] = 'green'
        individual['white'] = np.nan
        individual['green'] = 12
        individual['yellow'] = 18
        individual['red'] = 21
        individual['black'] = np.nan

        env.exog_med_transitions_evac = \
            pd.concat([env.exog_med_transitions_evac, \
                                                pd.DataFrame(individual, index = [0])],
                                                ignore_index = True)

    expected_result = {'delta_hat_e_k' : {'white' : 0, 'green' : 0, \
                                          'yellow' : 2, 'red' : 0, \
                                            'black' : 0}, \
                        'delta_hat_s_k' : {'white' : 0, 'green' : 0, \
                                           'yellow' : 0, 'red' : 0, \
                                            'black' : 0}
    }

    result = env._compute_delta_hat_k(tau_hat_k, decision)

    assert expected_result == result
                            
def test_compute_delta_hat_k_3(initial_state, seed):
    """Test the _compute_delta_hat_k method when the event is that individuals
    are to be loaded onto the ship.

    Test the _compute_delta_hat() method when the event is that 
    individuals are to be loaded onto the ship (e_k == 2). Note that 
    method returns both delta_hat_e_k and delta_hat_s_k, and calls several
    of the environment's other methods based on the state variable e_k.
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set tau_hat_k to 12 hours and the decision to null, i.e., no individual 
    # moves.
    tau_hat_k = 12
    decision = {'x_hl_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0},
                'x_sl_k' : {'white' : 0, 'green' : 10, 'yellow' : 0, 'red' : 0},
                'x_su_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}
    }

    env.state['e_k'] = 2

    # Remove the rows from the exog_med_transitions data frame and add
    # specific rows for the test case
    env.exog_med_transitions_evac.drop(env.exog_med_transitions_evac.index, inplace = True)

    for _ in range(12):
        individual = {}
        individual['arrival_time'] = 0
        individual['category'] = 'green'
        individual['white'] = np.nan
        individual['green'] = 12
        individual['yellow'] = 18
        individual['red'] = 21
        individual['black'] = np.nan

        env.exog_med_transitions_evac = \
            pd.concat([env.exog_med_transitions_evac, \
                                                pd.DataFrame(individual, index = [0])],
                                                ignore_index = True)

    # Note : when individuals are loaded onto the ship there is an element of 
    # randomness introduced as the transition times between medical triage
    # categories are sampled (within the _add_individuals method), and thus
    # we are uncertain as to the distribution of individuals by triage category
    # on the ship. However, we can state that for this test the number in the 
    # white and green categories must equal the number that is loaded onto the
    # ship (which is 10) and that the yellow, red, and black categories must 
    # have zero individuals.

    expected_result = {'white' : 0, 'green' : 0, \
                        'yellow' : 2, 'red' : 0, \
                        'black' : 0}
    
    result = env._compute_delta_hat_k(tau_hat_k, decision)

    assert expected_result == result['delta_hat_e_k']
    assert result['delta_hat_s_k']

def test_transition_fn(initial_state, seed):
    """Test the transition function that computes S_{k + 1}.

    Test the transition function that computes S_{k + 1} = S^M(S_k, x_k, 
    W_{k + 1}).
    """

    env = mass_evacuation.MassEvacuation(initial_state = initial_state, \
                                         seed = seed, default_rng = False)

    # Set the decision
    decision = {}

    # Set the state variable S_k for the test
    env.state = {'tau_k' : 24, 'e_k' : 3, \
                 'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, \
                              'red' : 5, 'black' : 10}, \
                'rho_s_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, \
                             'red' : 5, 'black' : 0}
    }

    # Set the exognous information
    exog_info = {'tau_hat_k' : 16, \
                 'e_hat_k' : 2, \
                 'delta_hat_e_k' : {'white' : 0, 'green' : 5, 'yellow' : 5, \
                              'red' : 0, 'black' : 15}, \
                 'delta_hat_s_k' : {'white' : 15, 'green' : 5, 'yellow' : 5, \
                             'red' : 0, 'black' : 0}
    }

    expected_result = {'tau_k' : 40, 'e_k' : 2, \
                 'rho_e_k' : {'white' : 0, 'green' : 5, 'yellow' : 5, \
                              'red' : 0, 'black' : 15}, \
                'rho_s_k' : {'white' : 15, 'green' : 5, 'yellow' : 5, \
                             'red' : 0, 'black' : 0}
    }

    assert expected_result == env._transition_fn(decision, exog_info)

                
