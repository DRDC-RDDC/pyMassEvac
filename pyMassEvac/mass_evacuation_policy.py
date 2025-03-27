"""
Decision policy module.

This module provide a class that implements the set of decision policies used
in Rempel (2024), see Section 4.3.

Throughout this module, the documentation will refer to Rempel (2024). For
a full description of the scenario and the sequential decision problem in
which these decision policies are used, see Section 3 and Section 4
respectively. In addition, the notation used throughout the code follows that
laid out in Powell (2022), Chapter 9.

The reference for the Rempel (2024) and Powell (2022) are as follows.

M. Rempel, "Modelling a major maritime disaster scenario using the   
universal modelling framework for sequential decisions", Safety 
Science, vol. 171, 106379, 2024.

W. Powell, "Reinforcemnt learning and stochastic optimization: A
unified framework for sequential decisions", Wiley, Hoboken, New, 
Jersey, 2022.
"""

import numpy as np

class MassEvacuationPolicy:
    """This class provides the set of decision poliices used in Rempel (2024).

    This class provides the set of decision policies used in Rempel (2024).
    The set of decision policies (see Section 4.3 of Rempel (2024), and Table 
    4) are used across the three decisions in x_k --- helicopter loading 
    (`x_hl_k`), ship loading (`x_sl_k`), and ship unloading (`x_su_k`). Using
    Powell's decision policy classes, each policy in this module is an
    instance of a policy function approximation and, in general, is given as 
    x_k = X^\\pi(S_k).

    Note: Although helicopters and ships are referred to throughout this code,
    they can be any transport vehicle and asset that can provide medical
    assistance at the evacuation site respectively. 

    Attributes
    ----------
    seed : int
        Integer that is the seed for all random number generators used in
        the decision policies; defualt `None`.
    """

    def __init__(self, default_rng = True, seed = None):
        """Initialize a decision policy object.

        The policy class contains the set of policy function
        approximations that were explored in Rempel (2024) [1]:
        green-first loading policy, yellow-first loading policy,
        critical-first loading policy, random loading / unloading
        policy, white-tags only unloading policy. See Table 4
        in Rempel (2024).

        In addition, a do nothing policy is implemented.
        """

        if default_rng:
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            else:
                self.rng = np.random.default_rng()
        else:
            if seed is not None:
                self.rng = np.random.RandomState(seed)
            else:
                self.rng = np.random.RandomState()

        return

    def do_nothing(self):
        """This policy returns a decision to do nothing.

        This policy returns a decision to do nothing. It may be
        used for testing purposes, or in a situation in
        which the helicopter or ship effectively do not make a
        decision.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # This decision policy does nothing
        decision = {'white': 0,
                    'green': 0, 
                    'yellow': 0, 
                    'red' : 0}        

        return decision

    def green_first_loading_policy(self, state, params):
        """Green-first loading policy.

        This method implements the green-first loading policy, see
        Section 4.3.1 of Rempel (2024). It prioritizes selecting
        individuals in triage categories as follows: green, white, 
        red, yellow.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See mass_evacuation.py for details.
        params : dict
            Policy parameters that provide constraints on the 
            decision: `total_capacity` is the total capacity available 
            at the location where the individuals will be loaded;
            `individual_capacity` is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        if state['e_k'] != 3:
            # Extract the policy parameters
            total_capacity = params['total_capacity']
            individual_capacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            num_loaded = 0
            num_available_total = 0
            num_available = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    num_available[k] = state['rho_e_k'][k]
                    num_available_total += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            # capacity = info_tuple['capacity']
            min_individual_capacity_needed = min({key: value for key, value \
                                                  in individual_capacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:

                # loading a ship - subtract the space that is consumed onboard 
                # the ship to calculate the remaining available capacity
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        total_capacity -= state['rho_s_k'][k] * \
                            individual_capacity[k]

            capacity_consumed = 0
            while (capacity_consumed < total_capacity) and (num_loaded < num_available_total) and (min_individual_capacity_needed <= total_capacity - capacity_consumed):

                # check if an individual with a green medical condition is available
                # to be loaded
                if num_available['green'] > 0:

                    if (capacity_consumed + individual_capacity['green']) <= total_capacity:
                        decision['green'] += 1
                        num_loaded += 1
                        capacity_consumed += individual_capacity['green']
                        num_available['green'] -= 1

                elif num_available['white'] > 0:

                    if (capacity_consumed + individual_capacity['white']) <= total_capacity:
                        decision['white'] += 1
                        num_loaded += 1
                        capacity_consumed += individual_capacity['white']
                        num_available['white'] -= 1

                elif num_available['red'] > 0:

                    if (capacity_consumed + individual_capacity['red']) <= total_capacity:
                        decision['red'] += 1
                        num_loaded += 1
                        capacity_consumed += individual_capacity['red']
                        num_available['red'] -= 1

                elif num_available['yellow'] > 0:

                    if (capacity_consumed + individual_capacity['yellow']) <= total_capacity:
                        decision['yellow'] += 1
                        num_loaded += 1
                        capacity_consumed += individual_capacity['yellow']
                        num_available['yellow'] -= 1

                min_individual_capacity_needed = min({key: value for \
                                                      key, value in individual_capacity.items() if key != 'black' and num_available[key] != 0}.values(), default = 0)  

        return decision

    def yellow_first_loading_policy(self, state, params):
        """Yellow-first loading policy.

        This method implements the yellow-first loading policy. It
        prioritizes selecting individuals in triage categories as
        follows: yellow, white, green, red.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See mass_evacuation.py for details.
        params : dict
            Policy parameters that provide constraints on the 
            decision: `total_capacity` is the total capacity available 
            at the location where the individuals will be loaded;
            `individual_capacity` is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        if state['e_k'] != 3:
            # Extract the policy parameters
            total_capacity = params['total_capacity']
            individual_capacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            num_loaded = 0
            num_available_total = 0
            num_available = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    num_available[k] = state['rho_e_k'][k]
                    num_available_total += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            min_individual_capacity_needed = min({key: value for key, value \
                                                  in individual_capacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:

                # loading a ship - subtract the space that is consumed onboard 
                # the ship to calculate the remaining available capacity
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        total_capacity -= state['rho_s_k'][k] * individual_capacity[k]

            capacity_consumed = 0
            while (capacity_consumed < total_capacity) and \
                (num_loaded < num_available_total) and \
                      (min_individual_capacity_needed <= total_capacity - \
                       capacity_consumed):

                # check if an individual with a yellow medical condition is available
                # to be loaded
                if (num_available['yellow'] > 0) & (capacity_consumed + \
                        individual_capacity['yellow'] <= total_capacity):

                    decision['yellow'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['yellow']
                    num_available['yellow'] -= 1

                elif (num_available['white'] > 0) & (capacity_consumed + individual_capacity['white'] <= total_capacity):

                    decision['white'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['white']
                    num_available['white'] -= 1

                elif (num_available['green'] > 0) & (capacity_consumed + individual_capacity['green'] <= total_capacity):

                    decision['green'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['green']
                    num_available['green'] -= 1

                elif (num_available['red'] > 0) & (capacity_consumed + individual_capacity['red'] <= total_capacity):

                    decision['red'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['red']
                    num_available['red'] -= 1

                min_individual_capacity_needed = min({key: value for key, value in individual_capacity.items() if key != 'black' and num_available[key] != 0}.values(), default = 0)                    

        return decision

    def critical_first_loading_policy(self, state, params):
        """Critical-first loading policy.

        This method implements the critical-first loading policy. It
        prioritizes selecting individuals in triage categories as
        follows: red, yellow, green, white.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See mass_evacuation.py for details.
        params : dict
            Policy parameters that provide constraints on the 
            decision: `total_capacity` is the total capacity available 
            at the location where the individuals will be loaded;
            `individual_capacity` is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        if state['e_k'] != 3:

            # Extract the policy parameters
            total_capacity = params['total_capacity']
            individual_capacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            num_loaded = 0
            num_available_total = 0
            num_available = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    num_available[k] = state['rho_e_k'][k]
                    num_available_total += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            min_individual_capacity_needed = min({key: value for key, value in individual_capacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:
                    
                # loading a ship
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        total_capacity -= state['rho_s_k'][k] * individual_capacity[k]

            capacity_consumed = 0
            while (capacity_consumed < total_capacity) and (num_loaded < num_available_total) and (min_individual_capacity_needed <= total_capacity - capacity_consumed):

                # check if an individual with a yellow medical condition is 
                # available to be loaded
                if (num_available['red'] > 0) & (capacity_consumed + individual_capacity['red'] <= total_capacity):

                    decision['red'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['red']
                    num_available['red'] -= 1

                elif (num_available['yellow'] > 0) & (capacity_consumed + individual_capacity['yellow'] <= total_capacity):

                    decision['yellow'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['yellow']
                    num_available['yellow'] -= 1

                elif (num_available['green'] > 0) & (capacity_consumed + individual_capacity['green'] <= total_capacity):

                    decision['green'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['green']
                    num_available['green'] -= 1

                elif (num_available['white'] > 0) & (capacity_consumed + individual_capacity['white'] <= total_capacity):

                    decision['white'] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity['white']
                    num_available['white'] -= 1

                min_individual_capacity_needed = min({key: value for key, value in individual_capacity.items() if key != 'black' and num_available[key] != 0}.values(), default = 0)                    

        return decision

    def random_loading_policy(self, state, params):
        """Random loading policy.

        This method implements the random loading policy. It
        does not prioritize individuals based on medical triage category,
        rather it randomly selects individuals.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See gym_mass_evacuation.mass_evacuation for details.
        params : dict
            Policy parameters that provide constraints on the 
            decision: `total_capacity` is the total capacity available 
            at the location where the individuals will be loaded;
            `individual_capacity` is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # Extract the policy parameters
        total_capacity = params['total_capacity']
        individual_capacity = params['individual_capacity']

        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        # check the max number of individuals that are available to be loaded
        num_loaded = 0
        num_available = 0

        for k in state['rho_e_k'].keys():
            if k != 'black':
                num_available += state['rho_e_k'][k]

        min_individual_capacity_needed = min(individual_capacity.values())

        # get the capacity of the helicopter / ship
        if state['e_k'] == 2:

            # loading a ship
            for k in state['rho_s_k'].keys():
                if k != 'black':
                    total_capacity -= state['rho_s_k'][k] * \
                        individual_capacity[k]

        capacity_consumed = 0
        while (capacity_consumed < total_capacity) and \
            (num_loaded < num_available) and (min_individual_capacity_needed <= total_capacity - capacity_consumed):

            # randomly select one of the four categories from which to load an
            # individual, but only from those categories from which there are 
            # people remaining to select
            l = list()
            for k in decision.keys():
                if (state['rho_e_k'][k] - decision[k]) > 0:
                    l.append(k)

            if len(l) > 0:

                # update the maxIndividualCapacity
                min_individual_capacity_needed = min({k: individual_capacity[k] for k in set(l) & set(individual_capacity.keys())}.values())

                r = self.rng.choice(l)

                if (capacity_consumed + individual_capacity[r]) <= total_capacity:
                    decision[r] += 1
                    num_loaded += 1
                    capacity_consumed += individual_capacity[r]

        return decision

    def random_unloading_policy(self, state, params):
        """Random unloading policy.

        This method implements the random unloading policy. It
        does not prioritize individuals based on medical triage category,
        rather it randomly selects individuals.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See gym_mass_evacuation.mass_evacuation for details.
        params : dict
            Policy parameters that provide constraints on the 
            decision: `num_to_unload` is the total number of individuals
            that will be selected to be unloaded.

        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # Extract the policy parameters
        num_to_unload = params['numToUnload']

        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        # Determine how many individuals are currently onboard the ship
        num_on_ship = sum(state['rho_s_k'].values())

        # The number of individuals to unload from the ship is the minimum of
        # the parameter passed to this policy and the actual number of 
        # individuals onboard the ship
        num_to_unload = min(num_to_unload, num_on_ship)

        for _ in range(num_to_unload):

            # randomly select one of the four categories from which to unload an
            # individual, but only from those categories from which there are 
            # people remaining to select

            # First, find the set of keys for which there are individuals on the
            # ship
            l = list()
            for k in decision.keys():
                if (state['rho_s_k'][k] - decision[k]) > 0:
                    l.append(k)

            # Second, randomly select one of those individuals
            if len(l) > 0:
                r = self.rng.choice(l)
                decision[r] += 1

        return decision

    def white_unloading_policy(self, state):
        """White-tags only unloading policy.

        This method implements the white-tags only unloading policy, see
        Section 4.4.2 in Rempel (2024). It does not prioritize individuals 
        based on medical triage category, rather it selects only those 
        individuals in the white-tag category.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
            See gym_mass_evacuation.mass_evacuation for details.
        
        Returns
        -------
        decision : dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected in each 
            medical triage category: `white`, `green`, `yellow`,
            `red`. Each key-value pair represents the number of 
            individuals selected from a triage category, e.g., 
            `white` : 2, `green` : 3, etc.
        """

        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}

        decision['white'] = state['rho_s_k']['white']

        return decision
