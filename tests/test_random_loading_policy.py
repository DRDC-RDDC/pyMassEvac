import pytest
import numpy as np
from pyMassEvac import mass_evacuation_policy 

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

def test_random_loading_policy_1(seed):
    """Test the random loading policy when `e_k = 1`.

    Test the random loading policy when `e_k = 1`.

    Parameters
    ----------
    seed : int
        Seed as defined in the pytest.fixture.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed = seed)

    S_k = {'tau_k' : 0, 'e_k' : 1, \
           'rho_e_k' : {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0, 'black' : 5},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, \
                         'red' : 0, 'black' : 0}
    }

    params = {'total_capacity' : 10, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                        'red' : 3, 'black' : np.nan}
    }

    expected_result = {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0}

    assert expected_result == p.random_loading_policy(state = S_k, \
                                                      params = params)
    
def test_random_loading_policy_2(seed):
    """Test the random loading policy when e_k = 2.

    Test the random loading policy when e_k = 2.

    Parameters
    ----------
    seed : int
        Seed as defined in the pytest.fixture.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed = seed)

    S_k = {'tau_k' : 0, 'e_k' : 2, \
           'rho_e_k' : {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0, 'black' : 5},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, \
                         'red' : 0, 'black' : 0}
    }

    params = {'total_capacity' : 10, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                        'red' : 3, 'black' : np.nan}
    }

    expected_result = {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0}

    assert expected_result == p.random_loading_policy(state = S_k, \
                                                      params = params)    
    
def test_random_loading_policy_3(seed):
    """Test the random loading policy when e_k = 2 and the ship is at capacity.

    Test the random loading policy when e_k = 2 and the ship is at capacity.

    Parameters
    ----------
    seed : int
        Seed as defined in the pytest.fixture.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed = seed)

    S_k = {'tau_k' : 0, 'e_k' : 2, \
           'rho_e_k' : {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0, 'black' : 5},
            'rho_s_k' : {'white' : 0, 'green' : 5, 'yellow' : 5, \
                         'red' : 0, 'black' : 0}
    }

    params = {'total_capacity' : 10, \
              'individual_capacity' : {'white' : 1, 'green' : 1, 'yellow' : 3, \
                        'red' : 3, 'black' : np.nan}
    }

    expected_result = {'white' : 0, 'green' : 0, 'yellow' : 0, \
                        'red' : 0}

    assert expected_result == p.random_loading_policy(state = S_k, \
                                                      params = params)
