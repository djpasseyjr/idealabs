#testPert.py
import svPerturb as Per
import numpy as np
import control as tf
from scipy import linalg as la
from matplotlib import pyplot as plt

"""Functions for testing the discrete time perterbation python code"""

def testLinkPer(N=300):
    theta = np.array([1]*100,complex)
    theta.imag = np.linspace(0,2*np.pi,100)
    theta = np.exp(theta)

    gainP = [0]*100
    gainG = [0]*100

    Success = [0]*6
    Y1 = []
    Y2 = []
    X = []
    for n in range(6): X = X + [n]*N

    for n in range(6):
        count = 0
        y1 = [0]*N
        y2 = [0]*N

    #Generate random transfer functions of size N

        for i in range(N):
            a = .3*np.random.rand()
            b = .3*np.random.rand()
            c = .3*np.random.rand()
            G = tf.tf([1.,c],[1.,-1.*complex(a,b)])

            for k in range(n):
                a = .5*np.random.rand()
                b = .5*np.random.rand()
                c = .5*np.random.rand()
                G = G*tf.tf([1.,c],[1.,-1.*complex(a,b)])
    
    #Perturb each transfer function and compute the infinity norm of the 
    #transfer function and it's perturbation

            Gp = G*Per.attack(G)
            for j in range(100):
                gainG[j] = la.norm(tf.evalfr(G,theta))
                gainP[j] = la.norm(tf.evalfr(Gp,theta))
            try:
                y1[i] = max(gainG)
            except ValueError:
                y1[i] = -10
            try:
                y2[i] = max(gainP)
            except ValueError:
                y2[i] = -10

            if np.isclose(y2[i],1):
                count += 1
        Y1 = Y1 + y1
        Y2 = Y2 + y2
        Success[n] = count

    plt.scatter(X,Y1)
    plt.scatter(X,Y2)
    plt.show()
    return Success,X,Y1,Y2

   
