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

def test_do_nothing(seed):
    """Test the do_nothing policy.

    Test whether the do_nothing policy returns a dict that
    performs no actions.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy(seed)
    decision = p.do_nothing()

    # assert
    assert decision == {'white' : 0, 'green' : 0, 'yellow' : 0, \
                        'red' : 0}
