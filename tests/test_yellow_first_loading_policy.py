import pytest
from gym_mass_evacuation import mass_evacuation_policy 
    
def test_yellowFirstLoadingPolicy_1():
    """Test the yellow-first loading policy when e_k == 1.

    Test whether the yellow-first loading policy returns a dict that
    prioritizes those individuals in the yellow-first triage state,
    followed by white, green, and then red.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy()

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 1, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}
    }

    # Set the policy parameters
    params = {'total_capacity' : 10, 
             'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }
    
    # Set the expected return value
    expected_value = {'white' : 1, 'green' : 0, 'yellow' : 3, 'red' : 0, \
                      'black' : 0}
    
    decision = p.yellowFirstLoadingPolicy(S_k, params)

    assert decision == expected_value

def test_yellowFirstLoadingPolicy_2():
    """Test the yellow-first loading policy when e_k == 2 and there is no
    available capacity onboard the ship.

    Test whether the yellow-first loading policy returns a dict that
    prioritizes those individuals in the yellow-first triage state,
    followed by white, green, and then red.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy()

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
    expected_value = {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                      'black' : 0}
    
    decision = p.yellowFirstLoadingPolicy(S_k, params)

    assert decision == expected_value

def test_greenFirstLoadingPolicy_3():
    """Test the yellow-first loading policy when e_k == 2 and there is 
    available capacity onboard the ship.

    Test whether the yellow-first loading policy returns a dict that
    prioritizes those individuals in the yellow-first triage state,
    followed by white, green, and then red.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy()

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 2, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                        'black' : 0}
    }

    # Set the policy parameters
    params = {'total_capacity' : 20, 
             'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }
    
    # Set the expected return value
    expected_value = {'white' : 5, 'green' : 0, 'yellow' : 5, 'red' : 0, \
                      'black' : 0}
    
    decision = p.yellowFirstLoadingPolicy(S_k, params)

    assert decision == expected_value

def test_yellowFirstLoadingPolicy_4():
    """Test the yellow-first loading policy when e_k == 3.

    Test whether the yellow-first loading policy returns a dict that states
    no individuals will be loaded as e_k == 3 is requesting the ship to be
    unloaded.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy()

    # Set the state variable S_k for the test
    S_k = {'tau_k' : 0, 'e_k' : 3, \
           'rho_e_k' : {'white' : 10, 'green' : 5, 'yellow' : 5, 'red' : 5, \
                        'black' : 10}, \
           'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                        'black' : 0}
    }

    # Set the policy parameters
    params = {'total_capacity' : 20, 
             'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                                      'red' : 3}
             }
    
    # Set the expected return value
    expected_value = {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, \
                      'black' : 0}
    
    decision = p.yellowFirstLoadingPolicy(S_k, params)

    assert decision == expected_value