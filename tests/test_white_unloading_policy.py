import pytest
import numpy as np
from gym_mass_evacuation import mass_evacuation_policy 

@pytest.fixture
def seed():
    
    return 49871037

def test_white_unloading_policy(seed):
    """Test the white_unloading_policy.

    Test the white unloading policy.

    Parameters
    ----------
    seed : int
        Seed as defined in the pytest.fixture.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed = seed)

    S_k = {'tau_k' : 0, 'e_k' : 3, \
           'rho_e_k' : {'white' : 5, 'green' : 5, 'yellow' : 0, \
                        'red' : 0, 'black' : 5},
            'rho_s_k' : {'white' : 12, 'green' : 0, 'yellow' : 0, \
                         'red' : 0, 'black' : 0}
    }

    expected_result = {'white' : 12, 'green' : 0, 'yellow' : 0, \
                        'red' : 0, 'black' : 0}

    assert expected_result == p.white_unloading_policy(state = S_k)