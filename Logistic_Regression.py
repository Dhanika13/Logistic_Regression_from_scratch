import numpy as np 
import pandas as pd 

class LogisticRegression:
    """  
    Class for Logistic Regression

    Parameters
    ----------
    learning_rate : float, default=1e-3
        Value of learning rate to update the parameters.

    max_iter : int, default=10000
        Maximum number of iterations. The solver iterates until convergence or
        reach this number iterations.

    tol : float, default=1e-4
        Tolerance to break the iteration in gradient descent process.
    """ 
    def __init__(self, learning_rate=1e-3, max_iter=10000, tol=1e-4):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
         
    #Sigmoid method
    def sigmoid(self, z):
        """ 
        Sigmoid function: g(z) = 1 / (1 + exp(-z))
        
        Parameters
        ----------
        Z : array-like of shape (n_samples, n_features)
            The input data.
        """ 
        return 1 / (1 + np.exp(-z))

    def loss(self, y_true, y_pred):
        """ 
        Parameters
        ----------
        y_true : array-like of shape (n_samples)
            Ground truth (correct) labels.
        
        y_prob : array-like of shape (n_samples)
            Predicted probalities.

        Returns
        -------
        loss : float
            Value of loss from learning process.
        """ 
        epsilon = 1e-9 # small value to avoid input 0 to log
        
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1 - y_true) * np.log(1 - y_pred + epsilon)
        
        loss = -np.mean(y1 + y2)
        
        return loss

    def predict_proba(self, X):
        """ 
        Predict the probability for given X

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        
        Returns
        -------
        y_pred : array of shape (n_samples)
            The prediction from input data.
        """ 
        z = np.dot(X, self.coefs) + self.intercept
        y_pred = self.sigmoid(z)
        
        return y_pred

    def fit(self, X, y):
        """
        Fit the logistic regression from the training dataset

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data

        y : {array-like} of shape (n_samples)
            Target values 
        """
        n_samples, n_features = X.shape

        # Initialize coefficients and intercept
        self.coefs = np.zeros(n_features)
        self.intercept = 0

        self.losses = []

        # Gradient descent
        for i in range(self.max_iter):
            # Compute y_pred
            y_pred = self.predict_proba(X)

            # Compute loss
            self.losses.append(self.loss(y, y_pred))

            # Compute gradients 
            dz = y_pred - y 
            
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
            
            # Update coefficients and intercept
            self.coefs -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

            # Break if the the change of loss is very small
            if (i > 1):
                d_loss = abs(self.losses[-1] - self.losses[-2]) 
                if (d_loss < self.tol):
                    break