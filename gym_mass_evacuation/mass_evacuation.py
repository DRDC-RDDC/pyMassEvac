"""
This module provides a custom gymnasium environment that implements the 
multi-domain mass evacuation problem described in Rempel (2024). Specifically,
the environment models the transport of individuals from an evacuation site 
via one or more helicopters, while one or more ships remain at the site with
the aim to provide medical assistance.

Within this environment, as the scenario progresses through time there are 
three types of decisions that are to be made: (i) which individuals waiting at 
the evacuation site are to be loaded onto a helicopter for transportation to
a forward operating base; (ii) which individuals are to be loaded from the
evacuation site onto a ship in order to receive medical attention; and (iii) 
which individuals are to be unloaded from the ship and return to the evacuation 
site so that others may board the ship and receive medical attention. 

Between these decisions being made, the individuals' medical conditions
change due to a variety of factors. While at the evacuation site, an 
individual's condition will deteriorate, while onboard the ship their
condition will improve. Once loaded onto a helicopter, their condition
is assumed to remain stable, and they are considered saved. The objective
of the multi-domain mass evacuation problem given in this environment is 
then to save the greatest number of lives possible (see equation (9) in 
Rempel (2024)) by seeking good policies for the three types of decisions
described above.

Throughout this module, the documentation will refer to Rempel (2024). For
a full description of the scenario, see Section 3. For a description of the
sequential decision problem that is implemented in this environment, see
Section 4.

The reference for the Rempel (2024) is as follows.

M. Rempel, "Modelling a major maritime disaster scenario using the   
universal modelling framework for sequential decisions", Safety 
Science, vol. 171, 106379, 2024.
"""

import math
import copy
import random
import gymnasium as gym
import numpy as np
import pandas as pd

from gym_mass_evacuation.mutable_priority_queue import MutablePriorityQueue


class MassEvacuation(gym.Env):

    """A gymnasium environment of a multi-domain mass evacuation scenario.
    """
    
    def __init__(self, initial_state, \
                 seed = None, default_rng = True):
        """Returns a new MassEvacuation gymnasium environment object.

        Creates a new MassEvacuation gymnasium environment object. The 
        object contains the: 
        
        - initial state S_0;
        - state S_k, where k is an event index; 
        - observation space; 
        - action space; 
        - random number generator object; and 
        - queue that stores the event arrival and inter-arrival times. 
        
        In addition, the initial transition times of individuals between 
        medical triage states are computed. This information is deemed 
        exogenous and is not knowable by the decision maker when deciding who 
        to load on the helicopters, ships, or unload from ships. Copies of 
        these two data frames are created and used in the reset method.

        Note: In Rempel (2024), `seed` was set to 20180529 and `default_rng`
        was set to False.

        Parameters
        ----------
        initial_state : dict
            Initial state S_0 of the environment. By default, it is set to
            rempel_2024_initial_state that was used in Rempel (2024) with
            a helicopter arriving at 48 hours and a ship arriving zero hours
            after the individual arrive at the evacuation location.
        seed : int, optional
            Random seed, by default None
        default_rng : bool, optional
            True if the numpy default_rng is to be used, False if RandomState 
            is to be used, by default True

        """

        super(MassEvacuation, self).__init__()

        # Set the render mode to None
        self.render_mode = None

        # Check if the initial state contains the required keys in the
        # dictionary.
        required_keys = ('m_e', 'm_s', 'c_h', 'c_s', 'delta_h', 'delta_s', \
                         'eta_h', 'eta_sl', 'eta_su', 'tau_k', 'e_k', \
                            'rho_e_k', 'rho_s_k', 'initial_helo_arrival', \
                                'initial_ship_arrival')

        if set(required_keys).issubset(initial_state):

            # Define the initial state S_0 - see Table 5 in Rempel (2024).
            self.initial_state = initial_state
            
            # Define the random number generator. 
            if default_rng is True:
                self.rng = np.random.default_rng(seed)
            else:
                self.rng = np.random.RandomState(seed)

            # Initialize the priority queue. Note that the state variable does not
            # have access to this queue.
            self.queue = MutablePriorityQueue()    

            # Add the arrival of the helicopters and ships to the event queue. The # arrival times defined in the initial state S_0 are relative to the
            # arrival of the individuals at the evacuation site.
            for i in range(len(self.initial_state['initial_helo_arrival'])):
                self.queue.put(self.initial_state['initial_helo_arrival'][i], 1)

            for i in range(len(self.initial_state['initial_ship_arrival'])):
                self.queue.put(self.initial_state['initial_ship_arrival'][i], 2)

            # Update the queue such that the arrival times of the events are 
            # relative to each previous event.
            self.queue.setRelative()
            
            # Define the initial values of the state variable S_k.
            self.state = {
                'tau_k' : 0,
                'e_k' : 0,
                'rho_e_k' : self.initial_state['rho_e_k'],
                'rho_s_k' : self.initial_state['rho_s_k']
            }

            # Define the observation space - this is the state space of the 
            # environment. See the definition of the state variable
            # in equation (1).
            self.observation_space = gym.spaces.Dict(
                { 
                    'tau_k' : gym.spaces.Discrete(168),
                    'e_k' : gym.spaces.Discrete(3),
                    'rho_e_k' : gym.spaces.Box(0, \
                        sum(self.initial_state['rho_e_k'].values()), \
                        shape = (1,4), dtype = np.intc),
                    'rho_s_k' : gym.spaces.Box(0, \
                        sum(self.initial_state['rho_e_k'].values()), \
                        shape = (1,4), dtype = np.intc)
                }
            )

            # Define the action space - this is the set of decisions that can 
            # be taken. See the definition in section 4.1.2. Note that is 
            # definition does not include the constraints on the decisions; 
            # those constraints (equations (2), (3), (5), and (6) would need to # be handled in the decision policies for loading a helicopter, 
            # loading the ship, and unloading the ship.
            self.action_space = gym.spaces.Dict(
                {
                    'x_hl_k' : gym.spaces.Box(0, \
                        self.initial_state['c_h'], \
                        shape = (1,4), \
                        dtype = np.intc),
                    'x_hs_k' : gym.spaces.Box(0, \
                        self.initial_state['c_s'], \
                        shape = (1,4), \
                        dtype = np.intc),
                    'x_su_k' : gym.spaces.Box(0, \
                        self.initial_state['c_s'], \
                        shape = (1,4), 
                        dtype = np.intc)
                }
            )

            # Create a data frame that stores the exogenous medical transition 
            # times for all individuals that are at the evacuation site or 
            # onboard the ship. Note that this information is not part of the 
            # observation, but is part of the enivronment.
            self.exog_med_transitions_evac = pd.DataFrame(columns = 
                                                        ['arrival_time',
                                                        'category',
                                                        'white',
                                                        'green',
                                                        'yellow',
                                                        'red',
                                                        'black'])
            
            self.exog_med_transitions_ship = pd.DataFrame(columns = 
                                                        ['arrival_time',
                                                        'category',
                                                        'white',
                                                        'green',
                                                        'yellow',
                                                        'red',
                                                        'black'])

            # Add the initial individuals to the self.exog_med_transitions_evac
            # data frame and randomly select their transition times between 
            # medical triage categories.
            self._add_individuals(initial_state['rho_e_k'], 'evac')

            # Make a copy of each exogenous data frame. These are used in the 
            # reset method to set the environment to its initial state.
            self.initial_med_transitions_evac = copy.deepcopy( \
                self.exog_med_transitions_evac)
            self.initial_med_transitions_ship = copy.deepcopy( \
                self.exog_med_transitions_ship)

            return

        else:
            raise KeyError(f'Keys "{required_keys}" does not exist')


    def _compute_reward(self, action):
        """Compute the contribution (or reward) for taking an action.

        The contribution function C(S_k, x_k) is the immediate reward 
        received when taking an action. See equation (8) in Rempel (2024).
        A contribution (or reward) is only recieved when a decision is made
        to load a helicopter, i.e., `self.state[e_k]` = 1, and the reward 
        received is equal to the number of individuals loaded onto the
        helicopter.

        Parameters
        ----------
        action : dict
            A dict of three actions: (i) load a helicopter `x_hl_k`; 
            (ii) load a ship `x_sl_k`; and (iii) unload a ship `x_su_k`. 
            The value of each of these keys is itself a dict with key-value 
            pairs that represent the number of individuals selected from the 
            `white`, `green`, `yellow`, and `red` triage categories. Note that
            individuals in the `black` triage category (deceased) are not 
            loaded onto a helicopter.

            For example, `action['x_hl_k'] = {'white' : 5, 'green' : 2,
            'yellow' : 1, 'red' : 0}`.

        Returns
        -------
        reward : int
            The contribution (or reward) received when loading a helicopter,
            i.e., the number of individuals loaded onto the helicopter; zero
            otherwise.
        """

        # Define the default reward
        reward = 0

        # Define the contribution function - see equation (8).
        if self.state['e_k'] == 1:
            reward = sum(action['x_hl_k'].values())

        return reward

    def observation(self):
        """Get the current state S_k.

        Returns
        -------
        dict
            The current state of the environment S_k as defined in equation (1) 
            of Rempel (2024). The state consists of four components: the kth 
            event `e_k`; the current system time `tau_k`; `rho_e_k`, the number 
            of individuals in each triage category `t` at the evacuation site;
            and `rho_s_k`, the number of individuals in each triage category `t`
            onboard the ship.
        """

        return self.state

    def reset(self, seed = None, options = None):
        """Reset the environment.

        Reset the environment to its initial state. The reset can occur in 
        either one of two ways. First, such that a single scenario is used, 
        (`options[single_scenario] = True`) as in Rempel (2024); or second such 
        that a different randomly selected scenario is initiated. The 
        difference between these two options is the sampled inter-transition 
        times between medical categories for individuals. When 
        `optionssingle_scenario] = True`, the times generated in the `__init__` 
        method are used; when `options[single_sceanrio] = False`, new times are 
        sampled.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default None. Not used in this implementation.
        options : dict
            A dict that consists of one option that is used to reset the 
            environment: `single_scenario`. 
            
            If `single_scenario` is `True`, then the sampled medical category 
            transition times given in the __init__ function are used to reset 
            the environment; if `False`, new transition times are sampled;
            by default False.

        Returns
        -------
        observation : dict
            Initial state S_k of the reset environment. The state consists of 
            four components: the kth event `e_k`; the current system time 
            `tau_k`; `rho_e_k`, the number of individuals in each triage 
            category `t` at the evacuation site; and `rho_s_k`, the number of 
            individuals in each triage category `t` onboard the ship.
        info : None
            Required by gymnasium, but not used.
        """

        # Reset the state to its original values
        self.state = {
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0}
        }

        # Reset the queue and add the arrival time for the helicopters and 
        # ships
        self.queue = MutablePriorityQueue()

        for i in range(len(self.initial_state['initial_helo_arrival'])):
            self.queue.put(self.initial_state['initial_helo_arrival'][i], 1)

        for i in range(len(self.initial_state['initial_ship_arrival'])):
            self.queue.put(self.initial_state['initial_ship_arrival'][i], 2)

        # Set the arrival times in the queue as relative
        self.queue.setRelative()

        # If a single sceanrio is selected, reset the individual's medical
        # transition times to those generated in the __init__ method.
        if options['single_scenario'] is True:

            self.exog_med_transitions_evac = copy.deepcopy(self.initial_med_transitions_evac)
            self.exog_med_transitions_ship = copy.deepcopy(self.initial_med_transitions_ship)

        else:

            # Multi-sceanrio in terms of the sampled transition times, not the
            # initial number of individuals in each triage category at the 
            # evacuation site

            # Reset the exogenous information that describes transitions of the
            # individuals between medical triage levels
            self.exog_med_transitions_evac.drop( \
                self.exog_med_transitions_evac.index, inplace = True)

            self.exog_med_transitions_ship.drop( \
                self.exog_med_transitions_ship.index, inplace = True)

            # Add the original number of individuals to the evacution site
            self._add_individuals(self.state['rho_e_k'], 'evac')

        observation = self.state
        info = {}

        return observation, info

    def step(self, action):
        """Transition the environment to the next state.

        This method performs three steps. First, it computes the immediate 
        reward that is received if the decision taken was to load a 
        helicopter, i.e., `x_hl_k`. Second, it determines the exogenous 
        information `W_k_plus_one` that arrives after the decision was made 
        (see Section 4.1.3 of Rempel (2024)). Third, it implements the 
        transition to the next state `S_k_plus_one`.

        Parameters
        ----------
        action : dict
            A dict of three actions: (i) load a helicopter `x_hl_k`; 
            (ii) load a ship `x_sl_k`; and (iii) unload a ship `x_su_k`. 
            Each value is itself a dict with key-value pairs that 
            represent the number of individuals selected from the 
            `white`, `green`, `yellow`, and `red` triage categories.

            For example, `action['x_hl_k'] = {'white' : 5, 'green' : 2,
            'yellow' : 1, 'red' : 0}`.

        Returns
        -------
        next_state : dict
            The next state `S_k_plus_one`.
        reward : int
            The contribution received is the decision taken was to 
            load a helicopter, i.e., `x_hl_k`.
        terminated : boolean
            True if the environment was stopped, False otherwise.
        truncated : boolean
            True if the environment was stopped, False otherwise.
        info : dict
            Not used.

        Notes
        -----
        The step method calls the following non-public methods. 

            - Compute reward : _compute_reward()
            - Get exogenous information : _exog_info_fn()
            - Compute next state : _transition_fn()    

        In turn, _exog_info_fn() calls _compute_delta_hat_k(), which itself
        calls several non-public methods. Note that the the order the 
        non-public methods are called in _compute_delta_hat_k() are dependent 
        upon the state S_k.

            - Update medical conditions : _update_medical_condition()
            - Remove individuals : _remove_individuals()
            - Add individuals : _add_individuals()

        Note that _remove_individuals() and _add_individuals() take a specific 
        action, either `x_hl_k`, `x_sl_k`, or `x_su_k` as a parameter rather 
        than a dict that contains all three decisions
        """

        next_state = copy.deepcopy(self.state)
        reward = 0
        terminated = False
        truncated = False
        info = {}

        # compute the contribution function - see equation (8).
        reward = self._compute_reward(action)

        # Get the exogenous information
        W_k_plus_one = self._exog_info_fn(action)

        # Execute the transition function - see Section 4.1.3, 
        # S_k_plus_one = S^m(S_k, x_k, W_k_plus_one)
        next_state = self._transition_fn(action, W_k_plus_one)

        # check if there are no individuals remaining at the evacuation site and onboard the ship
        if ((next_state['rho_e_k']['white'] + \
              next_state['rho_e_k']['green'] + \
                next_state['rho_e_k']['yellow'] + \
                    next_state['rho_e_k']['red']) == 0) & \
                        ((next_state['rho_s_k']['white'] + \
                          next_state['rho_s_k']['green'] + \
                            next_state['rho_s_k']['yellow'] + \
                                next_state['rho_s_k']['red']) == 0):
            terminated = True

        return next_state, reward, terminated, truncated, info

    def close(self):
        """Close the mass evacuation environment.

        This method closes the mass evacuation environment. It performs
        no actions.
        """

        return

    def render(self):
        """Compute the render frames as specified by render_mode 
        during the initialization of the environment.

        This function does not perform any actions. No render is computed.
        """

        return

    def _add_individuals(self, action, location):
        """Add individuals to a location's exogenous information data frame.

        Given the action to add individuals to a location, either 'ship'
        (`x_sl_k`) or 'evac' (`x_su_k`), create new rows in the exogenous 
        information data frame and sample the individuals' transition time 
        between the allowed medical triage categories.

        When adding individuals to the 'ship', the data frame 
        self.exog_med_transitions_ship is updated. When adding individuals
        to the `evac` site, the data frame self.exog_med_transitions_evac
        is updated.

        Parameters
        ----------
        action : dict
            A dict that describes the number of individuals in each triage
            category that are being moved to a location. When location is
            'ship', action should be `x_k[x_sl_k]`; when location is 'evac',
            action should be `x_k[x_su_k]`.
        location : str
            Either 'evac' or 'ship'.
        """

        # Define the valid transitions between triage categories that are 
        # allowed to occur at the evacuatio location and onboard the ship.
        valid_transitions_evac = {'white' : ['white', 'green', 'yellow','red'], 
                                  'green' : ['green', 'yellow', 'red'], 
                                  'yellow' : ['yellow', 'red'], 
                                  'red' : ['red']}

        valid_transitions_ship = {'white': [],
                                  'green' : ['green'], 
                                  'yellow' : ['yellow', 'green'], 
                                  'red' : ['red', 'yellow', 'green']}

        # The action to add a person to a location is a dict, which
        # contains the number of individuals per triage category.
        for k, v in action.items():
            for _ in range(v):

                # Set the individual's arrival time at the location
                # and their triage category.
                individual = {}
                individual['arrival_time'] = self.state['tau_k']
                individual['category'] = k

                for l in self.initial_state['rho_e_k'].keys():
                    individual[l] = np.nan

                if location == 'evac':

                    # Sample the individual's transition times between medical
                    # categories if the person is being moved from the ship to
                    # the evacuation location or arriving at the evacuation site
                    # at the start of the scenario. The times are the sampled
                    # times that an individual departs from a given medical
                    # condition.

                    last_category = ''
                    for l in valid_transitions_evac[k]:

                        if last_category == '':
                            individual[l] = individual['arrival_time'] + \
                                  -self.initial_state['m_e'][l] * \
                                    math.log(1 - self.rng.uniform(0, 1))
                        else:
                            individual[l] = individual[last_category] + \
                                -self.initial_state['m_e'][l] * \
                                    math.log(1 - self.rng.uniform(0, 1))

                        last_category = l

                    # add individual to the exogenous data frame
                    if len(self.exog_med_transitions_evac.index) == 0:

                        self.exog_med_transitions_evac = \
                            pd.DataFrame(individual, index = [0])
                    else:
                        self.exog_med_transitions_evac = \
                            pd.concat([self.exog_med_transitions_evac, \
                            pd.DataFrame(individual, index = [0])], \
                            ignore_index = True)

                elif location == 'ship':

                    # Sample the individual's transition times between medical
                    # categories if the person is being moved from the ship to
                    # the evacuation location.
                    last_category = ''
                    for l in valid_transitions_ship[k]:

                        if last_category == '':
                            individual[l] = individual['arrival_time'] + \
                                -self.initial_state['m_s'][l] * \
                                    math.log(1 - self.rng.uniform(0, 1))
                        else:
                            individual[l] = individual[last_category] + \
                                -self.initial_state['m_s'][l] * \
                                    math.log(1 - self.rng.uniform(0, 1))
                    
                        last_category = l            

                    # add individual to the exogenous data frame
                    if len(self.exog_med_transitions_ship.index) == 0:

                        self.exog_med_transitions_ship = \
                            pd.DataFrame(individual, index = [0])
                    else:
                        self.exog_med_transitions_ship = \
                            pd.concat([self.exog_med_transitions_ship, \
                            pd.DataFrame(individual, index = [0])], \
                            ignore_index = True)

        # Reset the data frame indicies.
        self.exog_med_transitions_evac.reset_index()
        self.exog_med_transitions_ship.reset_index()

        return

    # Function to remove individuals from a location.
    def _remove_individuals(self, action, location):
        """Remove individuals from a location's exogenous information data frame.

        Given the action to remove individuals to a location, either 'ship'
        (`x_su_k`) or 'evac' (`x_hl_k` or `x_sl_k`), delete appropriate rows in 
        the exogenous information data frame.

        When removing individuals to the 'ship', the data frame 
        self.exog_med_transitions_ship is updated. When removing individuals
        to the `evac` site, the data frame self.exog_med_transitions_evac
        is updated.

        Parameters
        ----------
        action : dict
            A dict that describes the number of individuals in each triage
            category that are being removed a location. When location is
            'ship', action should be `x_k[x_su_k]`; when location is 'evac',
            action should be `x_k[x_hl_k]` or `x_k[x_sl_k]`.
        location : str
            Either 'evac' or 'ship'.
        """

        idx = list()
        for k, v in action.items():

            # get the set of indices in the exog_transitions data frame for
            # this location that match the key
            if location == 'evac':

                idx = self.exog_med_transitions_evac[ \
                    self.exog_med_transitions_evac['category'] == \
                        k].index.to_list()

            elif location == 'ship':

                idx = self.exog_med_transitions_ship[ \
                    self.exog_med_transitions_ship['category'] == \
                        k].index.to_list()

            if v < len(idx):
                idx = list(self.rng.choice(idx, v, replace = False))
            else:
                pass

            # remove selected individuals from the location
            if location == 'evac':
                self.exog_med_transitions_evac.drop(idx, axis = 0, \
                                                    inplace = True)
            elif location == 'ship':
                self.exog_med_transitions_ship.drop(idx, axis = 0, \
                                                    inplace = True)

        # reset the index for each medical transitions data frame
        self.exog_med_transitions_evac.reset_index()
        self.exog_med_transitions_ship.reset_index()

        return

    def _update_medical_condition(self, tau_hat_k, location):
        """Update the medical triage categories of individuals at a location.

        Given the time between when an action is taken and the next action,
        the medical conditions of individuals at a location change. This method
        updates the exogenous information data frames 
        self.exog_med_transitions_evac when `location == evac` and
        self.exog_med_transitions_ship when `location == ship`. This method
        then returns the exogenous information `delta_hat_e_k` or 
        `delta_hat_s_k`. See section 4.1.3 of Rempel (2024).

        Parameters
        ----------
        tau_hat_k : int
            Inter-transition time to the next event k + 1.
        location : str
            Either `evac` or `ship`.

        Returns
        -------
        dict
            Either `delta_hat_e_k` (if `location == evac`) or 
            `delta_hat_s_k` (if `location == ship`).
        """

        # define delta_hat_e_k and delta_hat_s_k, which are the change in the
        # number of individuals in each triage category at the evacuation site
        # and onboard the ship respectively
        delta_hat_e_k = {'white': 0, 'green': 0, \
                         'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, \
                         'yellow': 0, 'red': 0, 'black': 0}

        # create tuples for the number of individuals entering and leaving
        # each medical category
        valid_transitions_evac = [('white', 'green'), ('green', 'yellow'),
                                  ('yellow', 'red'), ('red', 'black')]

        valid_transitions_ship = [('red', 'yellow'), ('yellow', 'green'),
                                  ('green', 'white')]

        # evacuation site
        if location == 'evac':

            for index, row in self.exog_med_transitions_evac.iterrows():

                if row['category'] != 'black':
                    current_med_category = row['category']

                    final_med_category = ''
                    for (initial, final) in valid_transitions_evac:
                        if row[initial] <= (self.state['tau_k'] + tau_hat_k):
                            final_med_category = final

                        if final_med_category == '':
                            pass
                        else:
                            self.exog_med_transitions_evac.at[ \
                                index, 'category'] = final_med_category

            # compute the delta_hat_e_k values
            for category in delta_hat_e_k.keys():

                if category in list(self.exog_med_transitions_evac[ \
                    'category'].value_counts().index):
                    delta_hat_e_k[category] = \
                        self.exog_med_transitions_evac[ \
                            'category'].value_counts()[category]
                else:
                    delta_hat_e_k[category] = 0

            return delta_hat_e_k

        elif location == 'ship':

            for index, row in self.exog_med_transitions_ship.iterrows():

                current_med_category = row['category']

                final_med_category = ''
                for (initial, final) in valid_transitions_ship:
                    if row[initial] <= (self.state['tau_k'] + tau_hat_k):
                        final_med_category = final

                    if final_med_category == '':
                        pass
                    else:
                        self.exog_med_transitions_ship.at[index, 'category'] = final_med_category

            # compute the delta_hat_e_k values
            for category in delta_hat_s_k.keys():

                if category in list(self.exog_med_transitions_ship[ \
                    'category'].value_counts().index):
                    delta_hat_s_k[category] = \
                        self.exog_med_transitions_ship[ \
                            'category'].value_counts()[category]
                else:
                    delta_hat_s_k[category] = 0

            return delta_hat_s_k

    def _compute_delta_hat_k(self, tau_hat_k, action):
        """Compute the updated number of individuals in each triage category.

        This method computes the updated number of individuals in each triage
        category at the evacuation site and onboard the ship. These are the 
        first two components of the exogenous information vector `W_k_plus_one`,
        more specifically `delta_hat_e_plus_one` and `delta_hat_s_plus_one`. See
        equation (7) in Rempel (2024).

        These changes are based not only on the time to the next event k + 1
        as presented by `tau_hat_k`, but also the action that has been taken
        to load the helicopter (`x_hl_k`), load the ship (`x_sl_k`), or unload 
        the ship (`x_su_k`).

        Parameters
        ----------
        tau_hat_k : int
            Inter-transition time to the next event k + 1.
        action : dict
            A dict that describes the number of individuals in each triage
            category that are being acted upon. The dict contains three
            key-value pairs, namely `x_hl_k` (loading a helicopter); `x_sl_k`
            (loading a ship), and `x_su_k` (unloading a ship).  

        Returns
        -------
        dict
            A dictionary with two key-value pairs, `delta_hat_e_k` and
            `delta_hat_s_k` that represent the updated number of individuals
            in each medical condition at the evacuation site and onboard the
            ship.
        """
        
        # define delta_hat_e_k and delta_hat_s_k, which are the change in the
        # number of individuals in each triage category at the evacuation site
        # and onboard the ship respectively
        delta_hat_e_k = {'white': 0, 'green': 0, \
                         'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, \
                         'yellow': 0, 'red': 0, 'black': 0}
        
        # create data frames for the number of individuals entering and leaving
        # each medical category
        valid_transitions_evac = [('white', 'green'), ('green', 'yellow'), \
                                  ('yellow', 'red'), ('red', 'black')]

        valid_transitions_ship = [('red', 'yellow'), ('yellow', 'green'), \
                                  ('green', 'white')]

        # self.state.e_k = 0: all individuals have arrived at evacuation site
        # tau_hat_k is the inter-arrival time to the next event, so the decay
        # of indivduals medical condition at the evacuation site must be
        # determined
        if self.state['e_k'] == 0:

            # update medical condition of individuals at evac site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

        # loading a helicopter; in this situation individuals have been
        # extracted from the evacuation site already so the exogenous info is
        # only on the remaining individuals
        if self.state['e_k'] == 1:

            # remove individuals from evac site in exogenous information
            self._remove_individuals(action['x_hl_k'], 'evac')

            # update medical condition of individuals remaining at evac site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

            # update medical condition of individuals onboard the ship
            delta_hat_s_k = self._update_medical_condition(tau_hat_k, 'ship')

        # loading a ship; here individuals have been already loaded onto a
        # ship and their deterioration onboard the ship plus those already
        # onboard the ship must be considered
        if self.state['e_k'] == 2:

            # remove individuals from evac site in exogenous information
            self._remove_individuals(action['x_sl_k'], 'evac')

            # add individuals onboard exogenous information for the ship
            self._add_individuals(action['x_sl_k'], 'ship')

            # update medical condition of individuals remaining at the evac
            # site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

            # update medical condition of individuals onboard the ship
            delta_hat_s_k = self._update_medical_condition(tau_hat_k, 'ship')

        # unloading ship; here individuals are unloaded from the ship, so at
        # the evacuation site the exogenous information must consider those 
        # already at the site and those departed the ship, and for the ship is
        # must only consider those after departure
        if self.state['e_k'] == 3:

            # remove individuals from ship in exogenous information
            self._remove_individuals(action['x_su_k'], 'ship')

            # add individuals to evac site exogenous information
            self._add_individuals(action['x_su_k'], 'evac')

            # update medical condition of individuals at the evac site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

            # update medical condition of individuals onboard the ship
            delta_hat_s_k = self._update_medical_condition(tau_hat_k, 'ship')

        return {'delta_hat_e_k': delta_hat_e_k, 'delta_hat_s_k' : delta_hat_s_k}

    def _exog_info_fn(self, action):
        """Get the exogenous information that arrives after an action is taken.

        This method returns the W_k_plus_one vector that represents the 
        exogenous information that arrives after an action `x_k` is taken. See
        equation (7) in Rempel (2024). The exogenous information vector consists
        of four components:

        - updated number of individuals in each triage category at the evacuation site (delta_hat_e_k_plus_one);
        - updated number of individuals in each triage category on the ship (delta_hat_s_k_plus_one);
        - event which triggers the next state (e_hat_k_plus_one):
            - 0: all individuals have arrived at the evac site
            - 1: helo arrives at the evac site and is ready to load individuals
            - 2: ship is ready to load individuals
            - 3: ship is ready to unload individuals
        - the inter-transition time to the next event (tau_hat_k_plus_one)

        For readability in the code, k_plus_one is dropped from the variables 
        and instead only k is used.

        Parameters
        ----------
        action : dict
            A dict that describes the number of individuals in each triage
            category that are being acted upon. The dict contains three
            key-value pairs, namely `x_hl_k` (loading a helicopter); `x_sl_k`
            (loading a ship), and `x_su_k` (unloading a ship).   

        Returns
        -------
        dict
            A dict with four key-value pairs that represent the W_{k + 1}
            exogenous information vector. The four key-value pairs on
            delta_hat_e_k, delta_hat_s_k, e_hat_k, and tau_hat_k.
        """
        
        delta_hat_e_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        # determine the next event that will take place (e_hat_k)
        tau_hat_k, e_hat_k = self.queue.get()

        medical_transition = self._compute_delta_hat_k(tau_hat_k, action)

        # e_hat_k == 1: a helo will be arriving next at the evacuation site and
        # selecting who to load, so we need to get the updated information for
        # for evacuation site
        if e_hat_k == 1:

            # Add the event to the queue when the helicopter will return. 
            self.queue.put(self.initial_state['eta_h'], 1, setRelative = True)

        # e_hat _k == 2: a ship will be arriving next to load individuals
        if e_hat_k == 2:

            # Add the event to the queue when the individuals from the ship will
            # be checked to determine if they are to be removed from the ship
            # and returned to the evacuation site (making room for others to
            # board the ship and receive medical attention).
            self.queue.put(self.initial_state['eta_su'], 3, setRelative = True)

        # e_hat_k == 3: a ship will be unloading individuals next; schedule
        # when the next set will be loading
        if e_hat_k == 3:
            self.queue.put(self.initial_state['eta_sl'], 2, setRelative = True)

        # return the exogenous information
        assert delta_hat_s_k['yellow'] >= 0

        return {'delta_hat_e_k': medical_transition['delta_hat_e_k'], \
                 'delta_hat_s_k': medical_transition['delta_hat_s_k'], \
                 'e_hat_k': e_hat_k, 'tau_hat_k': tau_hat_k}


    def _transition_fn(self, action, exog_info):
        """Compute the next state.

        This method implements the transition function described in section
        4.1.3 of Rempel (2024) - S_k_plus_one = S^m(S_k, x_k, W_k_plus_one).

        Parameters
        ----------
        decision : dict
            A dict that describes the number of individuals in each triage
            category that are being moved to a location.    
        exog_info : dict
            Exogenous information that arrives after a decision is made. 
            Contains four key-value pairs delta_hat_e_k, delta_hat_s_k,
            e_hat_k, and tau_hat_k. 

        Returns
        -------
        dict
            Next state S_{k + 1}. Contains four key-value pairs: tau_k,
            rho_e_k, rho_s_k, and e_k. See equation (1) in Rempel (2024).
        """

        tau_k = self.state['tau_k'] + exog_info['tau_hat_k']

        e_k = exog_info['e_hat_k']

        rho_e_k = exog_info['delta_hat_e_k']
        rho_s_k = exog_info['delta_hat_s_k']

        return {'tau_k': tau_k, 'rho_e_k': rho_e_k, 'rho_s_k': rho_s_k, 'e_k': e_k}