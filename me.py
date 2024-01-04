#me.py
'''
This is the optimization for the  effetive mass formula 
Read it carelfully. 

If i have mistakes in formulas or constants you can change thier values

Please install the numpy and the scipy libraies using :

pip install numpy scipy

you must have python 12 installed to do so 
'''

import numpy as np
from scipy.optimize import minimize

# Function to calculate Me
def calculate_Me(theta):
    # Define the Jacobian matrix J
    J = np.array([
        [-0.39 * np.sin(theta[0] + theta[1]) - 0.4 * np.sin(theta[0]) - 0.156 * np.sin(theta[0] + theta[1] + theta[2]),
         -0.39 * np.sin(theta[0] + theta[1]) - 0.156 * np.sin(theta[0] + theta[1] + theta[2]),
         -0.156 * np.sin(theta[0] + theta[1] + theta[2])],
        [0.39 * np.cos(theta[0] + theta[1]) + 0.4 * np.cos(theta[0]) + 0.156 * np.cos(theta[0] + theta[1] + theta[2]),
         0.39 * np.cos(theta[0] + theta[1]) + 0.156 * np.cos(theta[0] + theta[1] + theta[2]),
         0.156 * np.cos(theta[0] + theta[1] + theta[2])],
        [1, 1, 1]
    ])

    # Define the mass matrix M based on the provided equations
    M = np.array([
        [1.6181795 + 0.0156 * np.cos(theta[0] + theta[1]) + 0.429 * np.cos(theta[1]) + 0.01521 * np.cos(theta[2]),
         0.219131225 + 0.0078 * np.cos(theta[0] + theta[1]) + 0.39 * np.cos(theta[1]) + 0.03471 * np.cos(theta[2]),
         0.003042 + 0.0078 * np.cos(theta[0] + theta[1]) + 0.0078 * np.cos(theta[2])],
        [0.003043 + 0.0078 * np.cos(theta[0] + theta[1]) + 0.2145 * np.cos(theta[1]) + 0.01521 * np.cos(theta[2]),
         0.0741817 + 0.007605 * np.cos(theta[2]),
         0.000001156 + 0.01526 * np.cos(theta[2])],
        [0.003042 + 0.0078 * np.cos(theta[0] + theta[1]) + 0.007605 * np.cos(theta[2]),
         0.0019845 + 0.0061425 * np.cos(theta[2]),
         0.004015]
    ])

  
    u = np.array([0, -1, 0])

    
    Me = 1 / np.dot(u.T, np.dot(J.T, np.dot(np.linalg.inv(M), np.dot(J, u))))

    return Me


def constraint(theta):
    return np.concatenate(([360 - theta[0], 360 - theta[1], 360 - theta[2]],
                           [theta[0] - 360, theta[1] - 360, theta[2] - 360]))

# Initial guess for joint angles
initial_theta = np.zeros(3)

# Minimize Me with respect to theta, subject to constraints
result = minimize(calculate_Me, initial_theta, constraints={'type': 'ineq', 'fun': constraint})

# Extract the optimized joint angles
optimized_theta = result.x

print("Optimized Joint Angles:", optimized_theta)
print("Minimum Me:", result.fun)
