# Import routines

import numpy as np
import math
import random
from itertools import product

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0,0)] + [(i,j) for i in range(m) for j in range(m) if i != j]
        # print(f'action space: {self.action_space}')
        self.state_space = [(i,j,k) for i in range(m) for j in range(t) for k in range(d)]
        self.state_init = random.choice(self.state_space)
        self.poisson_dist = [2, 12, 4, 7, 8]

        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        location, time, day = state
        
        state_encod = np.zeros(m + t + d, dtype=int)

        state_encod[location] = 1
        state_encod[m + time] = 1
        state_encod[m + t + day] = 1

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]

        requests = np.random.poisson(self.poisson_dist[location])
        # if location == 0:
        #     requests = np.random.poisson(2)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request

        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append((0,0))

        return possible_actions_index, actions

    def step(self, state, action, Time_matrix):
        reward = self.reward_func(state, action, Time_matrix)
        next_state = self.next_state_func(state, action, Time_matrix)

        return next_state, reward

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        if action == (0, 0):
            reward = C * (-1)
        else:
            location, time, day = state
            start_loc, end_loc = action

            cost = 0
            time_to_reach_start_location = 0

            if location != start_loc:
                # find time required to reach start location
                time_to_reach_start_location = Time_matrix[location][start_loc][time][day]

                #now to get actual start time, add this time to original time
                time = int(time + time_to_reach_start_location)

                # if new time > 23, its next day. add 1 to day and reset time at 24
                time, day = CabDriver.get_time_day(time, day)
                
            # time required to go from start_location to end_location with new time and day
            action_time = Time_matrix[start_loc][end_loc][time][day]

            # final cost = Cost_per_hour * ( action_time + time_to_reach_start_location )
            cost = C * ( action_time + time_to_reach_start_location )

            # reward = Reward_per_hour * action_time - cost
            reward = R * action_time - cost

        return reward

    @staticmethod
    def get_time_day(t, d):
        time = int(t)
        day = d
        # if new time > 23, its next day. add 1 to day and reset time at 24
        day = day + time // 24
        time = time % 24
        day = day % 7

        return time, day


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        location, time, day = state
        start_loc, end_loc = action
        next_state = state

        if action == (0, 0):
            # if action is (0,0) then add 1 to time
            time, day = CabDriver.get_time_day(time + 1, day)
            next_state = location, time, day
        else:
            # calculate time_to_reach_start_location
            time_to_reach_start_location = Time_matrix[location][start_loc][time][day]
            time, day = CabDriver.get_time_day(time + time_to_reach_start_location, day)

            # calculate action_time
            action_time = Time_matrix[start_loc][end_loc][time][day]
            time, day = CabDriver.get_time_day(time + action_time, day)

            next_state = end_loc, time, day

        return next_state

    def reset(self):
        return self.action_space, self.state_space, self.state_init
