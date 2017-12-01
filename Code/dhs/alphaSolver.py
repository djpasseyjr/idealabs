#!usr/bin/python
#alphaSolver.py
"""This file contains functions I used to learn about the all pass filter  when trying to create a stable discrete
   time perturbation. 
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
from svPerturb import allPassAlpha2

def evalAP(theta,a):
    return (1-np.exp(complex(0,theta))*a.conjugate())/(np.exp(complex(0,theta)) - a)

def allPassFMatrix(theta,phi,r):
    """Matrix representation of the same problem"""
    A = np.array([[(np.cos(phi)/r)-np.cos(theta),-1*np.sin(theta)-(np.sin(phi)/r)],
                  [(np.sin(phi)/r)-np.sin(theta),np.cos(theta)+(np.cos(phi)/r)]])
    b = np.array([(np.cos(theta+phi)/r)-(np.sin(theta+phi)/r) - 1, (np.sin(theta+phi)/r) + (np.cos(theta+phi)/r)])
    try:
        ans = la.solve(A,b)
    except Exception:
        ans = np.array([100,0])
    return ans

def graphAPAlphaSol(X=None,Y=None,Z=None,res=100,tol=1e-4,f="norm"):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if X is None:
        dom = np.linspace(0,4*np.pi,res)
        X,Y = np.meshgrid(dom,dom)
        m,n = X.shape
        Z = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                if f == "norm":
                    Z[i,j] = la.norm(allPassAlpha2(X[i,j],Y[i,j]))
                if f == "unitary":
                    Z[i,j] = la.norm(allPassAlpha2(X[i,j],Y[i,j])) < 1
                if f == "real":
                    Z[i,j] = allPassAlpha2(X[i,j],Y[i,j]).real
                if f == "imag":
                    Z[i,j] = allPassAlpha2(X[i,j],Y[i,j]).imag
                    
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
    #fig.suptitle("Where alpha is a unit vector")
    plt.show()
    return X,Y,Z

def findCoord(Graph,point,tol=1e-5):
    """Finds the coordinates where the graph is equal to one
       and returns n and m where (x,y) = (pi/n,pi/m) """
    
    x = Graph[0][np.isclose(point,Graph[2],rtol=tol)]
    y = Graph[1][np.isclose(point,Graph[2],rtol=tol)]

    x = 1/x*np.pi
    y = 1/y*np.pi

    piCoord = [(0,0)]*len(x)
    for i in range(len(x)):
        piCoord[i] = (x[i],y[i])

    return piCoord

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def graphAPPhase_theta0(X=None,Y=None,Z=None,res=400,window=[-10,10], theta=np.pi/4):

    """This function graphs the phase of the all pass filter over the complex 
       numbers on the graph (i*window x window). Note that we fix theta, then 
       vary alpha in complex plane and then take the angle of the all pass filter. 
       I observed a discontinuous line through the origin whose slope was a 
       function of theta.

       When theta = pi the line appears to be y=0, where y is equal to i*b for some b in R.
       When theta = pi/2 the line appears to be y=x where x is in R
       When theta = pi/3 the line appears to be y = (2/3)x
       When theta = pi/4 the line appears to be y = (1/2)x
    
       My best guess based on these observations is that there is a discontinuity along the
       line y = (2/pi)*theta*x
    """
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    if X is None:
        dom = np.linspace(window[0],window[1],res)
        X,Y = np.meshgrid(dom,dom)
        m,n = X.shape
        Z = np.zeros((m,n))
        for i in range(m):
            for j in range(n):
                Z[i,j] = np.angle(evalAP(theta,complex(X[i,j],Y[i,j])))
                    
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
    ax.set_xlabel('Imaginary Axis')
    ax.set_ylabel('Real Axis')
    plt.show()
    return X,Y,Z

def graphAPPhase_alpha0(alpha=None,X=None,Y=None,lines=5,r=1):
    """ This function examines the change in the phase of the alpass filter
       with a fixed alpha and a variable theta. 

    It turns out that the length of alpha plays a huge role in the rate that
    of the filter changes wrt theta. When norm(alpha) = 1, the phase of the filter is 
    constant. As norm(alpha) -> inf, the phase of the filter becomes more variable with 
    alpha. The phase of alpha seems to act as a phase shift on the phase of the allpass 
    filter.

    """
    dom = np.linspace(-np.pi,np.pi,lines)

    if alpha is None:
        alpha = r*(np.cos(dom) + 1j*np.sin(dom))
        

    if X is None:
        X = np.linspace(-np.pi,np.pi,300)
        n = len(X)

        fig = plt.figure()
        ax = plt.subplot(111)

        for a in range(lines):
            Y = np.zeros(n)
            for j in range(n):
                Y[j] = np.angle(evalAP(X[j],alpha[a]))
            ax.plot(X,Y,label="$a = %s$" %str(np.round(alpha[a],6)))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          ncol=3)
    ax.set_title("Phase of AllPass($\theta$,a0) Over theta")
    plt.show()
 
    return alpha
