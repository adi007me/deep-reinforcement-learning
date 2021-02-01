import numpy as np

from Env import CabDriver

cabDriver = CabDriver()
Time_matrix = np.load("TM.npy")

test_action_space = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2), (1, 3), (1, 4), (2, 0), (2, 1), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)]
test_state_space = [(i,j,k) for i in range(5) for j in range(24) for k in range(7)]

assert cabDriver.action_space == test_action_space
assert cabDriver.state_space == test_state_space
assert cabDriver.state_init

assert list(cabDriver.state_encod_arch1((1, 16, 2))) == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

assert cabDriver.requests((2, 16, 2))[1]
assert cabDriver.requests((2, 7, 0))[0]

assert cabDriver.reward_func((1, 10, 0), (0,0), Time_matrix) == -5
assert cabDriver.reward_func((1, 10, 0), (2,3), Time_matrix) == -38
assert cabDriver.reward_func((2, 20, 0), (2,3), Time_matrix) == 12

assert cabDriver.next_state_func((2, 23, 0), (0, 0), Time_matrix) == (2, 0, 1)
assert cabDriver.next_state_func((1, 10, 0), (2, 3), Time_matrix) == (3, 23, 0)
assert cabDriver.next_state_func((1, 20, 0), (2, 3), Time_matrix) == (3, 7, 1)


arr = np.array([0,0,0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
print(arr.reshape(36,1))