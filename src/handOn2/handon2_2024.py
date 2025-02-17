import time

import numpy as np
import pymc as pm
from numpy import dtype

# k = 6
# n = 8
# n_simulations = 100000

# dice_rolls = np.random.randint(1, 7, (n_simulations, k * n))
# has_1 = (dice_rolls == 1).any(axis=1)
# non_single_1 = ~has_1
# prob_non_single_one = np.mean(non_single_1)
# print(f"Simulated probability of non-single '1's: {prob_non_single_one:.5f}")

# with pm.Model() as model:
#     total_ones = pm.Binomial('total_ones', n=k*n, p=1/6, shape=n_simulations)
#     prior = pm.sample_prior_predictive()
#
# total_ones_results = prior.prior['total_ones']
# no_ones_results = (total_ones_results == 0)
# prob_no_ones = no_ones_results.mean()

num_nodes = 1000
node_uptime = 0.999
num_simulations = 10000

start_time = time.time()

with pm.Model() as model:
    node_states = pm.Bernoulli('node_states', p=node_uptime, size=(num_simulations, num_nodes))
    system_uptime = pm.math.prod(node_states, axis=1)
    mean_system_uptime = pm.Deterministic('mean_system_uptime', pm.math.mean(system_uptime))
    prior_checks = pm.sample_prior_predictive()

end_time = time.time()
duration = end_time - start_time

#print in minutes and seconds
duration_gmt = time.gmtime(duration)
duration = time.strftime("%H:%M:%S", duration_gmt)
print(f"Duration: {duration}")

node_states_samples = prior_checks.prior['node_states'].values
node_states_samples = np.squeeze(node_states_samples)
nodes_up_count = np.sum(node_states_samples, axis=1)

system_uptime_samples = prior_checks.prior['mean_system_uptime'].values
overall_mean_system_uptime = np.mean(system_uptime_samples)
