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

def test_random_unloading_policy(seed):
    """Test the random unloading policy.

    Test the random loading policy.

    Parameters
    ----------
    seed : int
        Seed as defined in the pytest.fixture.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed = seed)

    S_k = {'tau_k' : 0, 'e_k' : 3, \
           'rho_e_k' : {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0, 'black' : 5},
            'rho_s_k' : {'white' : 10, 'green' : 0, 'yellow' : 0, \
                         'red' : 0, 'black' : 0}
    }

    params = {'numToUnload' : 10}

    expected_result = {'white' : 10, 'green' : 0, 'yellow' : 0, \
                        'red' : 0}

    assert expected_result == p.random_unloading_policy(state = S_k, \
                                                      params = params)