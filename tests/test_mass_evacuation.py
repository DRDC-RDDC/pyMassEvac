import pytest
from gym_mass_evacuation import mass_evacuation

def test_compute_reward_1():
    """Test the _compute_reward function when e_k = 1.

    Test the _compute_reward function when the state variable e_k is 1. This
    event is a helicopter loading event, and thus a decision should result in
    an immediate contribution being received. 
    """

    env = mass_evacuation.MassEvacuation()

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

def test_compute_reward_2():
    """Test the _compute_reward function when e_k = 2.

    Test the _compute_reward function when the state variable e_k is 2. This
    event is a ship loading event, and thus a decision should result in
    an immediate contribution being received. 
    """

    env = mass_evacuation.MassEvacuation()

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

def test_add_individuals_1():
    """Test the _add_individuals() method when individuals are to be added
    to the ship.

    Test the _add_individuals() method when individuals are to be added to 
    the ship. Note that _add_individuals() adds rows to the exog_med_transition_
    ship data frame, and does not change the state variable. The method does
    not check if space is available onboard the ship---it assumes that this 
    check has already been performed before this method is called.
    """
    env = mass_evacuation.MassEvacuation()

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
    
def test_add_individuals_2():
    """Test the _add_individuals() method when individuals are to be added
    to the evacuation site.

    Test the _add_individuals() method when individuals are to be added to 
    the evacuation site. Note that _add_individuals() adds rows to the 
    exog_med_transition_evac data frame, and does not change the state variable. 
    """
    env = mass_evacuation.MassEvacuation()

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

def test_remove_individuals_1():
    """Test the _remove_individuals() method when individuals are to be removed
    from the evacuation site.

    Test the _add_individuals() method when individuals are to be removed from 
    the evacuation site. Note that _remove_individuals() removes rows from the 
    exog_med_transition_evac data frame, and does not change the state variable. 
    """

    env = mass_evacuation.MassEvacuation(seed = 20180529, default_rng = False)

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

def test_remove_individuals_2():
    """Test the _remove_individuals() method when individuals are to be removed
    from the ship.

    Test the _add_individuals() method when individuals are to be removed from 
    the ship. Note that _remove_individuals() removes rows from the 
    exog_med_transition_ship data frame, and does not change the state variable. 
    """

    env = mass_evacuation.MassEvacuation(seed = 20180529, default_rng = False)

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
