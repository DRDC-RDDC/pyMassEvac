"""Decision policy module.

This module provide a class that implements the set of decision poliices used
in the recent article by Rempel in the journal Safety Science (vol. 171, 
106379, 2024). 

"""

import numpy as np

class MassEvacuationPolicy:
    """This class provides the set of decision poliices used in Rempel (2024). [1]_

    This class provides the set of decision policies used in Rempel (2024).[1]_
    The set of decision policies (see Section 4.3 of Rempel (2024), and Table 
    4) are used across the three decisions in `x_k` --- helicopter loading 
    (`x_hl_k`), ship loading (`x_sl_k`), and ship unloading (`x_su_k`). Using
    Powell's decision policy classes, each policy in this module is an
    instance of a policy function approximation and, in general, is given as 
    x_k = X^\\pi(S_k).

    Attributes
    ----------
    seed : int
        Integer that is the seed for all random number generators used in
        the decision policies.

    References
    ----------
    .. [1] M. Rempel, "Modelling a major maritime disaster scenario using the   
        universal modelling framework for sequential decisions", Safety 
        Science, vol. 171, 106379, 2024.
    """

    def __init__(self, seed):
        """Initialize a decision policy object.

        The policy class contains the set of policy function
        approximations that were explored in Rempel (2024) [1]_:
        green-first loading policy, yellow-first loading policy,
        critical-first loading policy, random loading / unloading
        policy, white-tags only unloading policy. See Table 4
        in Rempel (2024).

        In addition, a do nothing policy is implemented.

        """

        self.rng = np.random.default_rng(seed)

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
            medical triage category. Each key-value pair represents 
            the number of individuals selected from a triage 
            category, e.g., `white` : 2, `green` : 3, etc.
        """

        # This decision policy does nothing
        decision = {'white': 0,
                    'green': 0, 
                    'yellow': 0, 
                    'red': 0, 
                    'black': 0}        
       
        return decision

    def green_first_loading_policy(self, state, params):
        """Green-first loading policy.

        This method implements the green-first loading policy. It
        prioritizes selecting individuals in triage categories as
        follows: green, white, red, yellow.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: `tau_k`, `e_k`, `rho_e_k`, and `rho_s_k`.
        params : dict
            Policy parameters that provide constraints on the 
            decision: total_capacity is the total capacity available 
            at the location where the individuals will be loaded;
            individual_capacity is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        if (state['e_k'] != 3):
            # Extract the policy parameters
            totalCapacity = params['total_capacity']
            individualCapacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            numLoaded = 0
            numAvailableTotal = 0
            numAvailable = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    numAvailable[k] = state['rho_e_k'][k]
                    numAvailableTotal += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            # capacity = info_tuple['capacity']
            minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:

                # loading a ship - subtract the space that is consumed onboard 
                # the ship to calculate the remaining available capacity
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        totalCapacity -= state['rho_s_k'][k] * individualCapacity[k]

            capacityConsumed = 0
            while (capacityConsumed < totalCapacity) and (numLoaded < numAvailableTotal) and (minIndividualCapacityNeeded <= totalCapacity - capacityConsumed):

                # check if an individual with a green medical condition is available
                # to be loaded
                if numAvailable['green'] > 0:

                    if (capacityConsumed + individualCapacity['green']) <= totalCapacity:
                        decision['green'] += 1
                        numLoaded += 1
                        capacityConsumed += individualCapacity['green']
                        numAvailable['green'] -= 1

                elif numAvailable['white'] > 0:

                    if (capacityConsumed + individualCapacity['white']) <= totalCapacity:
                        decision['white'] += 1
                        numLoaded += 1
                        capacityConsumed += individualCapacity['white']
                        numAvailable['white'] -= 1

                elif numAvailable['red'] > 0:

                    if (capacityConsumed + individualCapacity['red']) <= totalCapacity:
                        decision['red'] += 1
                        numLoaded += 1
                        capacityConsumed += individualCapacity['red']
                        numAvailable['red'] -= 1

                elif numAvailable['yellow'] > 0:

                    if (capacityConsumed + individualCapacity['yellow']) <= totalCapacity:
                        decision['yellow'] += 1
                        numLoaded += 1
                        capacityConsumed += individualCapacity['yellow']
                        numAvailable['yellow'] -= 1

                minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black' and numAvailable[key] != 0}.values(), default = 0)                    

        return decision

    def yellowFirstLoadingPolicy(self, state, params):
        """Yellow-first loading policy.

        This method implements the yellow-first loading policy. It
        prioritizes selecting individuals in triage categories as
        follows: yellow, white, green, red.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: tau_k, e_k, rho_e_k, and rho_s_k.
        params : dict
            Policy parameters that provide constraints on the 
            decision: total_capacity is the total capacity available 
            at the location where the individuals will be loaded;
            individual_capacity is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        if (state['e_k'] != 3):
            # Extract the policy parameters
            totalCapacity = params['total_capacity']
            individualCapacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            numLoaded = 0
            numAvailableTotal = 0
            numAvailable = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    numAvailable[k] = state['rho_e_k'][k]
                    numAvailableTotal += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:

                # loading a ship - subtract the space that is consumed onboard 
                # the ship to calculate the remaining available capacity
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        totalCapacity -= state['rho_s_k'][k] * individualCapacity[k]

            capacityConsumed = 0
            while (capacityConsumed < totalCapacity) and (numLoaded < numAvailableTotal) and (minIndividualCapacityNeeded <= totalCapacity - capacityConsumed):

                # check if an individual with a yellow medical condition is available
                # to be loaded
                if (numAvailable['yellow'] > 0) & (capacityConsumed + individualCapacity['yellow'] <= totalCapacity):

                    decision['yellow'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['yellow']
                    numAvailable['yellow'] -= 1

                elif (numAvailable['white'] > 0) & (capacityConsumed + individualCapacity['white'] <= totalCapacity):

                    decision['white'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['white']
                    numAvailable['white'] -= 1

                elif (numAvailable['green'] > 0) & (capacityConsumed + individualCapacity['green'] <= totalCapacity):

                    decision['green'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['green']
                    numAvailable['green'] -= 1

                elif (numAvailable['red'] > 0) & (capacityConsumed + individualCapacity['red'] <= totalCapacity):

                    decision['red'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['red']
                    numAvailable['red'] -= 1

                minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black' and numAvailable[key] != 0}.values(), default = 0)                    

        return decision

    def criticalFirstLoadingPolicy(self, state, params):
        """Critical-first loading policy.

        This method implements the critical-first loading policy. It
        prioritizes selecting individuals in triage categories as
        follows: red, yellow, green, white.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: tau_k, e_k, rho_e_k, and rho_s_k.
        params : dict
            Policy parameters that provide constraints on the 
            decision: total_capacity is the total capacity available 
            at the location where the individuals will be loaded;
            individual_capacity is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        if (state['e_k'] != 3):

            # Extract the policy parameters
            totalCapacity = params['total_capacity']
            individualCapacity = params['individual_capacity']

            # check the number of individuals that are available to be loaded at
            # the evacuation site
            numLoaded = 0
            numAvailableTotal = 0
            numAvailable = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0}
            for k in state['rho_e_k'].keys():
                if k != 'black':
                    numAvailable[k] = state['rho_e_k'][k]
                    numAvailableTotal += state['rho_e_k'][k]

            # Compute the minimum capacity that can be loaded across the medical
            # conditions. This is required to cover an edge case in the policy.
            minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black'}.values())

            # get the capacity of the helicopter / ship
            if state['e_k'] == 2:
                    
                # loading a ship
                for k in state['rho_s_k'].keys():
                    if k != 'black':
                        totalCapacity -= state['rho_s_k'][k] * individualCapacity[k]

            capacityConsumed = 0
            while (capacityConsumed < totalCapacity) and (numLoaded < numAvailableTotal) and (minIndividualCapacityNeeded <= totalCapacity - capacityConsumed):

                # check if an individual with a yellow medical condition is available
                # to be loaded
                if (numAvailable['red'] > 0) & (capacityConsumed + individualCapacity['red'] <= totalCapacity):

                    decision['red'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['red']
                    numAvailable['red'] -= 1

                elif (numAvailable['yellow'] > 0) & (capacityConsumed + individualCapacity['yellow'] <= totalCapacity):

                    decision['yellow'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['yellow']
                    numAvailable['yellow'] -= 1

                elif (numAvailable['green'] > 0) & (capacityConsumed + individualCapacity['green'] <= totalCapacity):

                    decision['green'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['green']
                    numAvailable['green'] -= 1

                elif (numAvailable['white'] > 0) & (capacityConsumed + individualCapacity['white'] <= totalCapacity):

                    decision['white'] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity['white']
                    numAvailable['white'] -= 1

                minIndividualCapacityNeeded = min({key: value for key, value in individualCapacity.items() if key != 'black' and numAvailable[key] != 0}.values(), default = 0)                    

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
            key-value pairs: tau_k, e_k, rho_e_k, and rho_s_k.
        params : dict
            Policy parameters that provide constraints on the 
            decision: total_capacity is the total capacity available 
            at the location where the individuals will be loaded;
            individual_capacity is the amount of space each triage
            category consumes at the location.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """


        # Extract the policy parameters
        totalCapacity = params['total_capacity']
        individualCapacity = params['individual_capacity']

        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        # check the max number of individuals that are available to be loaded
        numLoaded = 0
        numAvailable = 0
        for k in state['rho_e_k'].keys():
            if k != 'black':
                numAvailable += state['rho_e_k'][k]

        minIndividualCapacityNeeded = min(individualCapacity.values())

        # get the capacity of the helicopter / ship
        if state['e_k'] == 2:

            # loading a ship
            for k in state['rho_s_k'].keys():
                if k != 'black':
                    totalCapacity -= state['rho_s_k'][k] * individualCapacity[k]

        capacityConsumed = 0
        while (capacityConsumed < totalCapacity) and (numLoaded < numAvailable) and (minIndividualCapacityNeeded <= totalCapacity - capacityConsumed):

            # randomly select one of the four categories from which to load an
            # individual, but only from those categories from which there are 
            # people remaining to select
            l = list()
            for k in decision.keys():
                if (k != 'black') and (state['rho_e_k'][k] - decision[k]) > 0:
                    l.append(k)

            if len(l) > 0:

                # update the maxIndividualCapacity
                minIndividualCapacityNeeded = min({k: individualCapacity[k] for k in set(l) & set(individualCapacity.keys())}.values())

                r = self.rng.choice(l)

                if (capacityConsumed + individualCapacity[r]) <= totalCapacity:
                    decision[r] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity[r]

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
            key-value pairs: tau_k, e_k, rho_e_k, and rho_s_k.
        params : dict
            Policy parameters that provide constraints on the 
            decision: numToUnload is the total number of individuals
            that will be selected to be unloaded.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """

        # Extract the policy parameters
        numToUnload = params['numToUnload']

        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        # Determine how many individuals are currently onboard the ship
        numOnShip = sum(state['rho_s_k'].values())

        # The number of individuals to unload from the ship is the minimum of
        # the parameter passed to this policy and the actual number of 
        # individuals onboard the ship
        numToUnload = min(numToUnload, numOnShip)

        for i in range(numToUnload):
            
            # randomly select one of the four categories from which to unload an
            # individual, but only from those categories from which there are 
            # people remaining to select

            # First, find the set of keys for which there are individuals on the
            # ship
            l = list()
            for k in decision.keys():
                if (k != 'black') and (state['rho_s_k'][k] - decision[k]) > 0:
                    l.append(k)

            # Second, randomly select one of those individuals
            if len(l) > 0:
                r = self.rng.choice(l)
                decision[r] += 1

        return decision

    def white_unloading_policy(self, state):
        """White-tags only unloading policy.

        This method implements the white-tags only unloading policy. It
        does not prioritize individuals based on medical triage category,
        rather it selects only those individuals in the white-tag category.

        Parameters
        ----------
        state : dict
            State of the mass evacuation problem as described in
            equation (1) in Rempel (2024). Dict contains four 
            key-value pairs: tau_k, e_k, rho_e_k, and rho_s_k.

        Returns
        -------
        dict
            Decision with four key-value pairs that describe the
            number of individuals that have been selected to take
            an action in each medical triage category.
        """


        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        decision['white'] = state['rho_s_k']['white']

        return decision