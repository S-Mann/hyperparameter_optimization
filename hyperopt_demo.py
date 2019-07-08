import numpy as np
import matplotlib.pyplot as plt
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import rand, tpe
from hyperopt import Trials
from hyperopt import fmin


def objective_function(x):
    """Objective function to minimize, for example 
    we have a polynomial 2(x^4)-2(x^3)-200(x^2)-5(x)-200"""
    
    ## Optimization Function
    # simple polynomial function
    f = np.poly1d([2,-2,-200,-5,-200])

    # returning a scaled value for the polynomial for certain value of x
    return f(x) * 0.05

# This part is to visualize the minimum
x = np.linspace(-10, 10, 10000)
y = objective_function(x)

# get the min value of y and corresponding x value
miny = min(y)
minx = x[np.argmin(y)]

# Visualize the function
plt.figure(figsize = (8, 6))
plt.style.use('fivethirtyeight')
plt.title('Objective Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.vlines(minx, min(y)- 50, max(y), linestyles = '-', colors = 'r')
plt.plot(x, y)

# Print out the minimum of the function and value
print('Minimum of %0.4f occurs at %0.4f' % (miny, minx))

# ------------------------

## Domain Space
space = hp.uniform('x', -10, 10)

samples = []

# Sample 10000 random values from the range, this is not need for the algorithm we just do this to visualize it.
for _ in range(10000):
    samples.append(sample(space))
    

# Histogram of the values
plt.hist(samples, bins = 30, edgecolor = 'black'); 
plt.xlabel('Domain Space Values'); plt.ylabel('Frequency of thr Values'); plt.title('Domain Space');


# Create objects for the algorithms
tpe_algo = tpe.suggest # Tree Structure Parzen Estimator
rand_algo = rand.suggest # Random Forest

# Create two trials objects to record your estimations
tpe_trials = Trials()
rand_trials = Trials()

# Run 2000 trials with the tpe algorithm with our objective function and domain space
tpe_best = fmin(fn=objective_function, space=space, algo=tpe_algo, trials=tpe_trials, 
                max_evals=2000, rstate= np.random.RandomState(50))

# Run 2000 trials with the random algorithm
rand_best = fmin(fn=objective_function, space=space, algo=rand_algo, trials=rand_trials, 
                 max_evals=2000, rstate= np.random.RandomState(50))

# Printing info on the trials
print('Minimum for TPE:        {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
print('Minimum for random:     {:.4f}'.format(rand_trials.best_trial['result']['loss']))
print('Actual minimum of f(x): {:.4f}'.format(miny))

# Print out information about number of trials
print('\nNumber of trials needed to attain minimum with TPE:    {}'.format(tpe_trials.best_trial['misc']['idxs']['x'][0]))
print('Number of trials needed to attain minimum with random: {}'.format(rand_trials.best_trial['misc']['idxs']['x'][0]))

# Print out information about value of x
print('\nBest value of x from TPE:    {:.4f}'.format(tpe_best['x']))
print('Best value of x from random: {:.4f}'.format(rand_best['x']))
print('Actual best value of x:      {:.4f}'.format(minx))
# -------------------------

