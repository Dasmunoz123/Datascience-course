import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def polynomial_regresion(x,y,X_train,X_test):
    
    ## para regresiones polinomiales se utilizan regresiones lineales con extensiones del dataset usando la función "PolynomialFeatures"
    
    poly3 = PolynomialFeatures(degree=3)
    poly6 = PolynomialFeatures(degree=6)
    poly9 = PolynomialFeatures(degree=9)

    x3 = poly3.fit_transform(x.reshape(-1,1))
    x6 = poly6.fit_transform(x.reshape(-1,1))
    x9 = poly9.fit_transform(x.reshape(-1,1))

    X_train_1, X_tests_1 = X_train.reshape(-1,1), X_test.reshape(-1,1) # order 1
    X_train_3, X_tests_3, y_train, y_test = train_test_split(x3,y,random_state = 0) # order 3
    X_train_6, X_tests_6, y_train, y_test = train_test_split(x6,y,random_state = 0) # order 6
    X_train_9, X_tests_9, y_train, y_test = train_test_split(x9,y,random_state = 0) # order 9

    regO1 = LinearRegression().fit(X_train_1, y_train)
    regO3 = LinearRegression().fit(X_train_3, y_train)
    regO6 = LinearRegression().fit(X_train_6, y_train)
    regO9 = LinearRegression().fit(X_train_9, y_train)

    ## Testing fitted models
    xspace = np.linspace(0,10,100)
    matrix = np.array((regO1.coef_ * xspace) + regO1.intercept_)

    auxiliar = [[regO3,3], [regO6,6], [regO9,9]]
    for aux in auxiliar:
        auxM = np.linspace(0,0,100) + aux[0].intercept_
        for i in range(1, aux[1] + 1): 
            auxM = auxM + aux[0].coef_[i] * np.power(xspace,i)

        matrix = np.vstack([matrix,auxM])

    return matrix
