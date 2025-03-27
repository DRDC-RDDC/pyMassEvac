"""Mutable priority queue module.

This module provides a class that implements a mutable priority queue. The
queue contains a list of upcoming events that will occur in the mass 
evacuation scenario.

"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class MutablePriorityQueue:
    """A mutable queue.
    
    This class provides a queue that can be changed overtime, and reordered
    given the addition of new items.

    Attributes
    ----------
    queue : DataFrame
        A pandas dataframe that consists of two columns: `tau_k` and `e_k`.
        The first is the time to the event, and the second is the 
        associated event.
    """

    def __init__(self):
        """Initialize a mutable queue.

        Initialize a pandas data frame with two columns: `tau_k` and `e_k`.
        The first is the time to the event, and the second is the 
        associated event.
        """

        self.queue = pd.DataFrame(columns = ['tau_k', 'e_k'])
        return

    def put(self, tau_k, e_k, setRelative = False):
        """Add a new event to the queue.
        
        Parameters
        ----------
        tau_k : int
            The system time the new event is set to occur.
        e_k : int        
            An int that identified the type of event that will occur.
        setRelative : boolean
            A boolean that indicates if the values in the tau_k column
            should updated to be relative; True (default) will set the
            values to be relative; False will not change the values.
        """

        if (len(self.queue.index) > 1) & (setRelative == True):
            # transform the tau_k's from relative into absolute so that they can
            # be sorted correctly
            for i in range(1, len(self.queue.index)):
                new_value = self.queue.loc[i, 'tau_k'] - \
                    self.queue.loc[i - 1, 'tau_k']
                self.queue.loc[i, 'tau_k'] = new_value                

        # add the event to the queue
        df = pd.DataFrame({'tau_k': tau_k, 'e_k': e_k}, index = [0])
        self.queue = pd.concat([self.queue, df], ignore_index = True)

        # sort the updated queue
        self.queue.sort_values(by = 'tau_k', axis = 0, ascending = True,
        inplace = True)

        # reset the indicies
        self.queue.reset_index(drop = True, inplace = True)

        if setRelative:
            self.setRelative()

        return

    def setRelative(self):
        """Update the queue such that the values in the tau_k column are
        relative.

        Update the queue such that the values in the tau_k column are relative.
        """

        # update the values of tau_k so that they are relative to the next
        # event that will arise
        for r in range(len(self.queue.index) - 1, 0, -1):
            new_value = self.queue.loc[r, 'tau_k'] - \
                self.queue.loc[r - 1, 'tau_k']
            self.queue.loc[r, 'tau_k'] = new_value

        return


    def get(self):
        """Get the next event in the queue.

        Get the next event in the queue.

        Returns
        -------
        tau_k : int
            The time the next event will occur.
        e_k : int
            An integer indicating the type of event that will next occur.
        """

        # get the highest priority event
        tau_k = self.queue['tau_k'][0]
        e_k = self.queue['e_k'][0]
        
        # remove the event from the queue
        self.queue.drop(0, axis = 0, inplace = True)

        self.queue.reset_index(drop = True, inplace = True)

        return tau_k, e_k
