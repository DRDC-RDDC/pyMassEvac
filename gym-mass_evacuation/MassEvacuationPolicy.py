"""
Mass evacuation benchmark decision policies

"""

import pandas as pd
import numpy as np
import math

class MassEvacuationPolicy():

    def __init__(self):
        """
        """

        return

    def do_nothing(self):

        # This decision policy does nothing
        decision = {'white': 0, 
                    'green': 0, 
                    'yellow': 0, 
                    'red': 0, 
                    'black': 0}        
        
        return decision

    def greenFirstLoadingPolicy(self, state, params):
        
        """
        this function implements the green-first policy for a helicopter
        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        # Extract the policy parameters
        totalCapacity = params['total_capacity']
        individualCapacity = params['individual_capacity']
    
        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

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

            # loading a ship - subtract the space that is consumed onboard the ship to calculate the remaining available
            # capacity
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

        """
        this function implements the yellow-first policy for a helo / ship
        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        # Extract the policy parameters
        totalCapacity = params['total_capacity']
        individualCapacity = params['individual_capacity']

        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

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

    def criticalFirst_loading_policy(self, state, params):
        """
        this function implements the critial-first policy for a helo / ship
        :param state: namedtuple - the state of the model at a given time
        :param info_tuple: tuple - contains the parameters needed to run the policy
        :return: a decision made based on the policy
        """

        # Extract the policy parameters
        totalCapacity = params['total_capacity']
        individualCapacity = params['individual_capacity']


        # define the default decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

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
                totalCapacity -= state.rho_s_k[k] * individualCapacity[k]

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
        if state['e_k'] == 1:

            # loading a ship
            for k in state.rho_s_k.keys():
                totalCapacity -= state.rho_s_k[k] * individualCapacity[k]

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

                r = self.model.rng.choice(l)

                if (capacityConsumed + individualCapacity[r]) <= totalCapacity:
                    decision[r] += 1
                    numLoaded += 1
                    capacityConsumed += individualCapacity[r]

        return decision

    def random_unloading_policy(self, state, params):

        # Extract the policy parameters
        numToUnload = params['numToUnload']

        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        # Determine how many individuals are currently onboard the ship
        numOnShip = 0
        for k in state['rho_s_k'].keys():
            numOnShip += state.rho_s_k[k]

        # The number of individuals to unload from the ship is the minimum of
        # the parameter passed to this policy and the actual number of 
        # individuals onboard the ship
        numToUnload = min(numToUnload, numOnShip)

        for i in range(numToUnload):
            
            # randomly select one of the four categories from which to unload an
            # individual, but only from those categories from which there are 
            # people remaining to select
            l = list()
            for k in decision.keys():
                if (k != 'black') and (state['rho_s_k'][k] - decision[k]) > 0:
                    l.append(k)

            if len(l) > 0:
                r = self.model.rng.choice(l)
                decision[r] += 1

        return decision

    def white_unloading_policy(self, state):

        # Initialize the decision
        decision = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        decision['white'] = state['rho_s_k']['white']

        return decision
