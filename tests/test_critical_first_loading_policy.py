import pytest
from gym_mass_evacuation import mass_evacuation_policy
    
@pytest.fixture
def seed():
    """Define a pytest fixture for the testing seed.

    Define a pytest fixture for the testing seed.

    Returns
    -------
    int
        Seed for use in random number generators.
    """

    return 49871037

def test_critical_first_loading_policy_1(seed):
    """Test the critical-first loading policy when `e_k = 1`.

    Test whether the critical-first loading policy returns a dict that
    prioritizes those individuals in the red-first triage state,
    followed by yellow, green, and then white.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 1, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}
    }

    # Set the policy parameters
    params = {'total_capacity' : 10, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }

    # Set the expected return value
    expected_value = {'white' : 0, 'green' : 1, 'yellow' : 0, 'red' : 3}

    decision = p.critical_first_loading_policy(S_k, params)

    assert decision == expected_value

def test_critical_first_loading_policy_2(seed):
    """Test the critical-first loading policy when `e_k = 2` and there is no
    available capacity onboard the ship.

    Test whether the critical-first loading policy returns a dict that
    prioritizes those individuals in the red-first triage state,
    followed by yellow, green, and then white.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 2, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 10, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                        'black' : 0}
    }

    # Set the policy parameters
    params = {'total_capacity' : 10, 
             'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }

    # Set the expected return value
    expected_value = {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}

    decision = p.critical_first_loading_policy(S_k, params)

    assert decision == expected_value

def test_critical_first_loading_policy_3(seed):
    """Test the critical-first loading policy when `e_k = 2` and there is 
    available capacity onboard the ship.

    Test whether the critical-first loading policy returns a dict that
    prioritizes those individuals in the red-first triage state,
    followed by yellow, green, and then white.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 2, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                        'black' : 0}
    }

    # Set the policy parameters
    params = {'total_capacity' : 20, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }
    
    # Set the expected return value
    expected_value = {'white' : 0, 'green' : 2, 'yellow' : 1, 'red' : 5}
    
    decision = p.critical_first_loading_policy(S_k, params)

    assert decision == expected_value

def test_critical_first_loading_policy_4(seed):
    """Test the critical-first loading policy when e_k == 3.

    Test whether the critical-first loading policy returns a dict that states
    no individuals will be loaded as e_k == 3 is requesting the ship to be
    unloaded.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed)

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 3, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                        'black' : 0}
    }

    # Set the policy parameters
    params = {'total_capacity' : 20, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }

    # Set the expected return value
    expected_value = {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0}

    decision = p.critical_first_loading_policy(S_k, params)

    assert decision == expected_value
