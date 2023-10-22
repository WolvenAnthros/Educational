import numpy as np
from geneticalgorithm import geneticalgorithm as ga
from SFQ_calc import  reward_calculation


varbound = np.array([[-1, 1]] * 125)

model = ga(function=reward_calculation, dimension=125, variable_type='int', variable_boundaries=varbound)

model.run()