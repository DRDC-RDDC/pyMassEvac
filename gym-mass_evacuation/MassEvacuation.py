import gymnasium as gym
import numpy as np
import pandas as pd
import copy
import math
import random

from MutablePriorityQueue import MutablePriorityQueue

"""
Define an environment for a mass evacuation scenario. The evacuation of 
individuals is performed via helicopter, but there is also a ship on site
which can assist in improving the medical condition of individuals at the
evacuation location.

Documentation to create a custom environment is provided at:
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
"""

class MassEvacuation(gym.Env):

    """
    Custom gymnasium environment to study the transport of individuals from an evacuation site to a forward operating location.
    """
    
    def __init__(self, seed = None, default_rng = True):
        """Initialization of a new MassEvacuation gymnasium environment.

        Creates a new MassEvacuation gymnasium environment object. The 
        object contains the initial state S_0, defines the state S_k, 
        observation space, action space, and the queue that stores the 
        event arrival and inter-arrival times. In addition, the initial 
        transition times of individuals between medical triage states are 
        computed. This information is deemed exogenous and is not knowable by 
        the decision maker when deciding who to load on the helicopters, ships, 
        or unload from ships.

        Note: In Rempel (2024), seed was set to 20180529 and default_rng
        was set to False.

        Parameters
        ----------
        seed : int, optional
            Random seed, by default None
        default_rng : bool, optional
            True if the numpy default_rng is to be used, False if RandomState is to be used, by default True
        """
        super(MassEvacuation, self).__init__()
        
        # Define the initial state S_0 - see Table 5 in Rempel (2024).
        self.initial_state = {
            'm_e' : {'white' : 120, 'green' : 48, 'yellow' : 8, 'red' : 1.5},
            'm_s' : {'green' : 48, 'yellow' : 72, 'red' : 120},
            'c_h' : 10,
            'c_s' : 50,
            'delta_h' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'delta_s' : {'white' : 1, 'green' : 1, 'yellow' : 3, 'red' : 3},
            'eta_h' : 3,
            'eta_sl' : 24,
            'eta_su' : 1,
            'tau_k' : 0,
            'e_k' : 0,
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0},
            'initial_helo_arrival' : [48],
            'initial_ship_arrival' : [0]
        }
        
        # Define the random number generator. 
        if default_rng == True:
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
            'rho_e_k' : {'white' : 0, 'green' : 475, 'yellow' : 20, 'red' : 5, 'black' : 0},
            'rho_s_k' : {'white' : 0, 'green' : 0, 'yellow' : 0, 'red' : 0, 'black' : 0}
        }

        # Define the observation space - this is the state space of the 
        # environment. See the definition of the state variable
        # in equation (1).
        self.observation_space = gym.spaces.Dict(
            { 
                'tau_k' : gym.spaces.Discrete(168),
                'e_k' : gym.spaces.Discrete(3),
                'rho_e_k' : gym.spaces.Box(0, 500, shape = (1,4), dtype = np.intc),
                'rho_s_k' : gym.spaces.Box(0, 500, shape = (1,4), dtype = np.intc)
            }
        )

        # Define the action space - this is the set of decisions that can be 
        # taken. See the definition in section 4.1.2. Note that is definition 
        # does not include the constraints on the decisions; those constraints 
        # (equations (2), (3), (5), and (6) would need to be handled in the 
        # decision policies for loading a helicopter, loading the ship, and 
        # unloading the ship.
        self.action_space = gym.spaces.Dict(
            {
                'x_hl_k' : gym.spaces.Box(0, 10, shape = (1,4), dtype = np.intc),
                'x_hs_k' : gym.spaces.Box(0, 50, shape = (1,4), dtype = np.intc),
                'x_su_k' : gym.spaces.Box(0, 50, shape = (1,4), dtype = np.intc)
            }
        )

        # Create a data frame that stores the exogenous medical transition 
        # times for all individuals that are at the evacuation site or onboard 
        # the ship. Note that this information is not part of the observation, 
        # but is part of the enivronment.
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
        self._add_individuals(self.state['rho_e_k'], 'evac')
        
        # Make a copy of each exogenous data frame. These are used in the 
        # reset method to set the environment to its initial state.
        self.initial_med_transitions_evac = copy.deepcopy(self.exog_med_transitions_evac)
        self.initial_med_transitions_ship = copy.deepcopy(self.exog_med_transitions_ship)

        return

    def _compute_reward(self, action):
        """Compute the contribution for taking an action.

        The contribution function C(S_k, x_k) is the immediate reward 
        received when making a decision. See equation (8) in Rempel (2024).

        Parameters
        ----------
        action : dict
            A dict of actions to load a helicopter 'x_hl_k', load a ship
            'x_sl_k', and unload a ship 'x_su_k'. Each is a dict of the number
            of individuals selected from the white, green, yellow, and
            red triage categories.

        Returns
        -------
        int
            The reward received when loading a helicopter, i.e., the number
            of individuals loaded onto the helicopter. 
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
            of Rempel (2024).
        """

        return self.state

    def reset(self, single_scenario = True):
        """Reset the environment.

        Reset the environment to its initial state. The reset can occur such 
        that a single scenario is used, as in Rempel (2024), or a different
        randomly scenario is initiated. The difference between these two 
        options is the sampled inter-transition times between medical categories
        for individuals. When single_scenario is True, the times generated in
        the __init__ method are used; when single_sceanrio is False, new times
        are sampled.

        Parameters
        ----------
        single_scenario : bool, optional
            If True, use the sampled medical category transition times given in the __init__ function; if False, sample new transition times, by default True.

        Returns
        -------
        observation : dict
            Initial state of the reset environment.
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
        if single_scenario:

            self.exog_med_transitions_evac = copy.deepcopy(self.initial_med_transitions_evac)
            self.exog_med_transitions_ship = copy.deepcopy(self.initial_med_transitions_ship)

        else:

            # Multi-sceanrio in terms of the sampled transition times, not the
            # initial number of individuals in each triage category at the 
            # evacuation site

            # Reset the exogenous information that describes transitions of the
            # individuals between medical triage levels
            self.exog_med_transitions_evac.drop(self.exog_med_transitions_evac.index, inplace = True)

            self.exog_med_transitions_ship.drop(self.exog_med_transitions_ship.index, inplace = True)
            
            # Add the original number of individuals to the evacution site
            self._add_individuals(self.state['rho_e_k'], 'evac')

        observation = self.state
        info = {}
        
        return observation, info

    def step(self, action):
        """Transition the environment to the next state.

        This method performs three steps. First, it computes the immediate 
        reward that is received if the decision taken was to load a helicopter.
        Second, it determines the exogenous information W_{k + 1} that arrives
        after the decision was made (see Section 4.1.3 of Rempel (2024)). Third,
        is implements the transition function S^m(S_k, x_k, W_{k + 1}) that 
        determines the next state.

        Parameters
        ----------
        action : dict
            A dict of actions to load a helicopter 'x_hl_k', load a ship
            'x_sl_k', and unload a ship 'x_su_k'. Each is a dict of the number
            of individuals selected from the white, green, yellow, and
            red triage categories.

        Returns
        -------
        nextState : dict
            The next state S_{k + 1}.
        reward : int
            The contribution received is the decision taken was to load a helicopter.
        terminated : boolean
            True if the environment was stopped, False otherwise.
        truncated : boolean
            True if the environment was stopped, False otherwise.
        info : dict
            Not used.
        """

        nextState = copy.deepcopy(self.state)
        reward = 0
        terminated = False
        truncated = False
        info = dict()

        # compute the contribution function - see equation (8).
        reward = self._compute_reward(action)

        # Get the exogenous information
        W_k_plus_one = self._exog_info_fn(action)
        
        # Execute the transition function - see Section 4.1.3, S_{k + 1} = S^m(S_k, x_k, W_{k + 1})
        nextState = self._transition_fn(action, W_k_plus_one)
        
        # check if there are no individuals remaining at the evacuation site and onboard the ship
        if (((nextState['rho_e_k']['white'] + nextState['rho_e_k']['green'] +
            nextState['rho_e_k']['yellow'] + nextState['rho_e_k']['red']) == 0) &
            ((nextState['rho_s_k']['white'] + nextState['rho_s_k']['green'] +
            nextState['rho_s_k']['yellow'] + nextState['rho_s_k']['red']) == 0)):
            terminated = True
                                                        
        return nextState, reward, terminated, truncated, info
    
    def close(self):
        pass

    def render(self, mode = None):

        return

    def _update_queue(self, tau_k, e_k):

        self.queue.put(tau_k, e_k, setRelative = True)

        return

    # Function to add individuals to a location, either the evacuation site 'evac' or the 'ship'.
    def _add_individuals(self, decision, location):

        """
        Add individuals to either the ship (from the evacuation site) or to the evacuation site (from the ship).
        """
        
        valid_transitions_evac = {'white' : ['white', 'green', 'yellow','red'], 
                                  'green' : ['green', 'yellow', 'red'], 
                                  'yellow' : ['yellow', 'red'], 
                                  'red' : ['red']}

        valid_transitions_ship = {'white': [],
                                  'green' : ['green'], 
                                  'yellow' : ['yellow', 'green'], 
                                  'red' : ['red', 'yellow', 'green']}

        for k, v in decision.items():
            for i in range(v):
                
                individual = dict()
                individual['arrival_time'] = self.state['tau_k']
                individual['category'] = k

                for l in self.initial_state['rho_e_k'].keys():
                    individual[l] = np.nan

                if (location == 'evac'):

                    last_category = ''
                    for l in valid_transitions_evac[k]:

                        if last_category == '':
                            individual[l] = individual['arrival_time'] + -self.initial_state['m_e'][l] * math.log(1 - self.rng.uniform(0, 1))
                        else:
                            individual[l] = individual[last_category] + -self.initial_state['m_e'][l] * math.log(1 - self.rng.uniform(0, 1))
                    
                        last_category = l

                    # add individual to the exogenous data frame
                    if len(self.exog_med_transitions_evac.index) == 0:

                        self.exog_med_transitions_evac = pd.DataFrame(individual, index = [0])
                    else:
                        self.exog_med_transitions_evac = pd.concat([
                            self.exog_med_transitions_evac,
                            pd.DataFrame(individual, index = [0])],
                            ignore_index = True)

                elif (location == 'ship'):

                    last_category = ''
                    for l in valid_transitions_ship[k]:

                        if last_category == '':
                            individual[l] = individual['arrival_time'] + -self.initial_state['m_s'][l] * math.log(1 - self.rng.uniform(0, 1))
                        else:
                            individual[l] = individual[last_category] + -self.initial_state['m_s'][l] * math.log(1 - self.rng.uniform(0, 1))
                    
                        last_category = l            

                    # add individual to the exogenous data frame
                    if len(self.exog_med_transitions_ship.index) == 0:

                        self.exog_med_transitions_ship = pd.DataFrame(
                            individual, index = [0])
                    else:
                        self.exog_med_transitions_ship = pd.concat([
                            self.exog_med_transitions_ship,
                            pd.DataFrame(individual, index = [0])],
                            ignore_index = True)

        self.exog_med_transitions_evac.reset_index()
        self.exog_med_transitions_evac.reset_index()

        return

    # Function to remove individuals from a location.
    def _remove_individuals(self, decision, location):

        idx = list()
        for k, v in decision.items():

            # get the set of indices in the exog_transitions data frame for
            # this location that match the key 
            if location == 'evac':

                idx = self.exog_med_transitions_evac[self.exog_med_transitions_evac['category'] == k].index.to_list()
            elif location == 'ship':

                idx = self.exog_med_transitions_ship[self.exog_med_transitions_ship['category'] == k].index.to_list()

            if (v < len(idx)):
                idx = list(self.rng.choice(idx, v, replace = False))
            else:
                pass

            # remove selected individuals from the location
            if location == 'evac':
                self.exog_med_transitions_evac.drop(idx, axis = 0,
                                                    inplace = True)
            elif location == 'ship':
                self.exog_med_transitions_ship.drop(idx, axis = 0,
                                                    inplace = True)

        # reset the index for each medical transitions data frame
        self.exog_med_transitions_evac.reset_index()
        self.exog_med_transitions_ship.reset_index()

        return

    def _update_medical_condition(self, tau_hat_k, location):
            
        # define delta_hat_e_k and delta_hat_s_k, which are the change in the
        # number of individuals in each triage category at the evacuation site
        # and onboard the ship respectively
        delta_hat_e_k = {'white': 0, 'green': 0, 
                         'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, 
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
                            self.exog_med_transitions_evac.at[index, 'category'] = final_med_category

            # compute the delta_hat_e_k values
            for category in delta_hat_e_k.keys():

                if category in list(self.exog_med_transitions_evac['category'].value_counts().index):
                    delta_hat_e_k[category] = self.exog_med_transitions_evac['category'].value_counts()[category]
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

                if category in list(self.exog_med_transitions_ship['category'].value_counts().index):
                    delta_hat_s_k[category] = self.exog_med_transitions_ship['category'].value_counts()[category]
                else:
                    delta_hat_s_k[category] = 0

            return delta_hat_s_k

    def _compute_delta_hat_k(self, tau_hat_k, decision):

        """
        This function computes the first and second component of the W_{k + 1} vector of exogenous information.
        """
        
        # define delta_hat_e_k and delta_hat_s_k, which are the change in the
        # number of individuals in each triage category at the evacuation site
        # and onboard the ship respectively
        delta_hat_e_k = {'white': 0, 'green': 0, 
                         'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, 
                         'yellow': 0, 'red': 0, 'black': 0}
        
        # create data frames for the number of individuals entering and leaving
        # each medical category
        valid_transitions_evac = [('white', 'green'), ('green', 'yellow'),
                                  ('yellow', 'red'), ('red', 'black')]

        valid_transitions_ship = [('red', 'yellow'), ('yellow', 'green'),
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
            self._remove_individuals(decision['x_hl_k'], 'evac')

            # update medical condition of individuals remaining at evac site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

            # update medical condition of individuals onboard the ship
            delta_hat_s_k = self._update_medical_condition(tau_hat_k, 'ship')  

        # loading a ship; here individuals have been already loaded onto a 
        # ship and their deterioration onboard the ship plus those already
        # onboard the ship must be considered
        if self.state['e_k'] == 2:

            # remove individuals from evac site in exogenous information
            self._remove_individuals(decision['x_sl_k'], 'evac')

            # add individuals onboard exogenous information for the ship
            self._add_individuals(decision['x_sl_k'], 'ship')

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
            self._remove_individuals(decision['x_su_k'], 'ship')

            # add individuals to evac site exogenous information
            self._add_individuals(decision['x_su_k'], 'evac')

            # update medical condition of individuals at the evac site
            delta_hat_e_k = self._update_medical_condition(tau_hat_k, 'evac')

            # update medical condition of individuals onboard the ship
            delta_hat_s_k = self._update_medical_condition(tau_hat_k, 'ship')  

        return {'delta_hat_e_k': delta_hat_e_k, 'delta_hat_s_k' : delta_hat_s_k}

    def _exog_info_fn(self, decision):

        """
        This function returns the W_{t + 1} vector that represents the 
        exogenous information that arrives after a decision is made. This consists of four components:

        - change in the number of individuals in each triage category at the evacuation site (delta_hat_e_k);
        - change in the number of individuals in each triage category on the ship (delta_hat_s_k);
        - event which triggers the next state (e_hat_k). these may be:
            - 0: all individuals have arrived at the evac site
            - 1: helo arrives at the evac site and is ready to load individuals
            - 2: ship arrives at the evac site and is ready to load individuals
            - 3: ship is ready to unload individuals
        - the inter transition time to the next event (tau_hat_k)
        """

        delta_hat_e_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}
        delta_hat_s_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        # determine the next event that will take place (e_hat_k)
        tau_hat_k, e_hat_k = self.queue.get()

        medical_transition = self._compute_delta_hat_k(tau_hat_k, decision)

        # e_hat_k == 1: a helo will be arriving next at the evacuation site and
        # selecting who to load, so we need to get the updated information for
        # for evacuation site
        if e_hat_k == 1:
            
            # Add the event to the queue when the helicopter will return. note # the 3 below is for a revisit time in three hours; this is 
            # hardcoded for now, but will be computed in the future based on the
            # distance from the forward operating loaction to the evacuate site
            self.queue.put(self.initial_state['eta_h'], 1, setRelative = True)

        # e_hat _k == 2: a ship will be arriving next to load individuals
        if e_hat_k == 2:
        
            # Add the event to the queue when the individuals from the ship will
            # be checked to determine if they are to be removed from the ship
            # and returned to the evacuation site (making room for others to
            # board the ship and receive medical attention). note that the 
            # number 24 below is an assumption and consistutes a policy decision
            self.queue.put(self.initial_state['eta_sl'], 3, setRelative = True)  

        # e_hat_k == 3: a ship will be unloading individuals next; schedule 
        # when the next set will be loading
        if e_hat_k == 3:
            self.queue.put(self.initial_state['eta_su'], 2, setRelative = True)

        # return the exogenous information
        assert delta_hat_s_k['yellow'] >= 0

        return {'delta_hat_e_k': medical_transition['delta_hat_e_k'],
                 'delta_hat_s_k': medical_transition['delta_hat_s_k'], 'e_hat_k': e_hat_k, 'tau_hat_k': tau_hat_k}

    # This function defines S_{k + 1} = S^M(S_k, X^Pi(S_k), W_{k + 1})
    def _transition_fn(self, decision, exog_info):

        tau_k = self.state['tau_k'] + exog_info['tau_hat_k']

        e_k = exog_info['e_hat_k']

        rho_e_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}
        rho_s_k = {'white': 0, 'green': 0, 'yellow': 0, 'red': 0, 'black': 0}

        rho_e_k = exog_info['delta_hat_e_k']
        rho_s_k = exog_info['delta_hat_s_k']

        return {'tau_k': tau_k, 'rho_e_k': rho_e_k, 'rho_s_k': rho_s_k, 'e_k': e_k}