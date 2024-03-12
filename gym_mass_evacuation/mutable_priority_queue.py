import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class MutablePriorityQueue():

    def __init__(self):

        self.queue = pd.DataFrame(columns = ['tau_k', 'e_k'])
        return

    def put(self, tau_k, e_k, setRelative = False):

        if (len(self.queue.index) > 1) & (setRelative == True):
            # transform the tau_k's from relative into absolute so that they can
            # be sorted correctly
            for i in range(1, len(self.queue.index)):
                self.queue.iloc[i]['tau_k'] += self.queue.iloc[i - 1]['tau_k']

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

        # update the values of tau_k so that they are relative to the next
        # event that will arise
        for r in range(len(self.queue.index) - 1, 0, -1):
            self.queue['tau_k'][r] -= self.queue['tau_k'][r - 1]

        #for r in range(1, len(self.queue.index)):
        #    self.queue['tau_k'][r] -= self.queue['tau_k'][r - 1]
            #self.queue.loc['tau_k', r] -= self.queue['tau_k'][r - 1]

        return


    def get(self):

        # get the highest priority event
        tau_k = self.queue['tau_k'][0]
        e_k = self.queue['e_k'][0]
        
        # remove the event from the queue
        self.queue.drop(0, axis = 0, inplace = True)

        self.queue.reset_index(drop = True, inplace = True)

        return tau_k, e_k
