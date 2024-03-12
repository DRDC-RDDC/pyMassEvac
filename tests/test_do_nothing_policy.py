import pytest
from gym_mass_evacuation import mass_evacuation_policy 

def test_do_nothing():
    """Test the do_nothing policy.

    Test whether the do_nothing policy returns a dict that
    performs no actions.
    """

    p = mass_evacuation_policy.MassEvacuationPolicy()
    decision = p.do_nothing()

    # assert
    assert decision == {'white' : 0, 'green' : 0, 'yellow' : 0, \
                        'red' : 0, 'black' : 0}