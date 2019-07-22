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

# ---------------------------------------------------
# This is a more practical example of using hyperopt
# ---------------------------------------------------

from io import StringIO
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from math import sqrt

dataset=pd.read_csv("dataset/dataset.csv")
dataset.head()

excluded_columns = [x not in ['model','msrp','cost_per_unit','profit_per_unit'] for x in dataset.columns]

X = dataset.iloc[:,excluded_columns].values
y = dataset.iloc[:,dataset.columns == 'msrp'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(X_train)
X_train = imp.transform(X_train)
imp = imp.fit(X_test)
X_test = imp.transform(X_test)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def knn_obj_func(n):    
    classifier = KNeighborsClassifier(n_neighbors=int(n))
    classifier.fit(X_train, np.ravel(y_train, order='C'))
    
    y_pred = classifier.predict(X_test)
    
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    # with k = 5 , rmse = 40507.76274692575
    return rmse

# This is a random k value and its RMSE
random_hyperparam = 7
random_loss = knn_obj_func(7)

space = hp.uniform('x', 2, 1000)
samples = []

for _ in range(30):
    samples.append(sample(space))    

# Create objects for the algorithms
tpe_algo = tpe.suggest # Tree Structure Parzen Estimator
rand_algo = rand.suggest # Random Forest

# Create two trials objects to record your estimations
tpe_trials = Trials()
rand_trials = Trials()

# Run 2000 trials with the tpe algorithm with our objective function and domain space
tpe_best = fmin(fn=knn_obj_func, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=100, rstate= np.random.RandomState(50))

# Run 2000 trials with the random algorithm with our objective function and domain space
rand_best = fmin(fn=knn_obj_func, space=space, algo=rand_algo, trials=rand_trials, max_evals=100, rstate= np.random.RandomState(50))

# Printing info on the trials
print('Minimum loss for TPE:        {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
print('Minimum loss for random:     {:.4f}'.format(rand_trials.best_trial['result']['loss']))
print('Default k\'s loss:            {:.4f}'.format(random_loss))

# Print out information about number of trials
print('\nNumber of trials needed to attain minimum with TPE:     {}'.format(tpe_trials.best_trial['misc']['idxs']['x'][0]))
print('Number of trials needed to attain minimum with random:  {}'.format(rand_trials.best_trial['misc']['idxs']['x'][0]))

# Print out information about value of x
print('\nBest value of k from TPE:    {:.0f}'.format(tpe_best['x']))
print('Best value of k from random: {:.0f}'.format(rand_best['x']))
print('Default value of k:          {:.0f}'.format(random_hyperparam))

plt.bar(['Default','TPE','Random Forest'], 
        [random_loss,tpe_trials.best_trial['result']['loss'],rand_trials.best_trial['result']['loss']], 
        edgecolor='black')
plt.xlabel('Algorithms')
plt.ylabel('RMSE')
plt.title('Loss for each Approach(Lower is Better)')
plt.axhline(y=tpe_trials.best_trial['result']['loss'], color = 'r')
plt.plot()
