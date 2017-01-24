import numpy as np

def predict_proba(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) for the given
    data with the given coefficients.
    '''
    linear_predictor = np.dot(X, coeffs)
    denom = 1 + np.exp(-linear_predictor)
    return 1.0 / denom

def predict(X, coeffs, thres=0.5):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with the given
    coefficients.
    '''
    probs = predict_proba(X, coeffs)
    return (probs >= thres).astype(int)

def log_likelihood(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the log likelihood of the data with the given coefficients.
    '''

    p = predict_proba(X, coeffs)
    cost_per_observation = y * np.log(p) + (1 - y) * np.log(1 - p)
    return np.sum(cost_per_observation) 

def log_likelihood_gradient(X, y, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the log likelihood at the given value for the
    coeffs. Return an array of the same size as the coeffs array.
    '''
    p = predict_proba(X, coeffs)  
    gradient = X.T.dot(y - p)
    return gradient

def accuracy(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUPUT: float

    Calculate the percent of predictions which equal the true values.
    '''
    return float(np.sum(y_true == y_pred)) / len(y_true)

def precision(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive predictions which were correct.
    '''
    return float(np.sum(y_true * y_pred)) / np.sum(y_pred)

def recall(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive cases which were correctly predicted.
    '''
    return float(np.sum(y_true * y_pred)) / np.sum(y_true)


class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept=False):
        '''
        INPUT: GradientAscent, function, function
        OUTPUT: None

        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''
        # Initialize coefficients in run method once you know how many features
        # you have.
        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept

    def run(self, X, y, alpha=0.1, num_iterations=100):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''
        ## add intercept (if we want to)
        if self.fit_intercept:
            X = self.add_intercept(X)

        ## Initialize coeffs to all zeros
        self.coeffs = np.zeros(X.shape[1])
        
        self.cost_history = []
        for i in range(num_iterations):
            grad = self.gradient(X, y, self.coeffs)
            self.coeffs = self.coeffs + alpha / X.shape[0] * grad
            cost = self.cost(X, y, self.coeffs)
            self.cost_history.append(cost)
            
    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's.
        '''
        if self.fit_intercept:
            X = self.add_intercept(X)
        return self.predict_func(X, self.coeffs)

    def add_intercept(self, X):
        '''
        INPUT: 2 dimensional numpy array
        OUTPUT: 2 dimensional numpy array

        Return a new 2d array with a column of ones added as the first
        column of X.
        '''
        return np.hstack((np.ones((X.shape[0], 1)), X))
    