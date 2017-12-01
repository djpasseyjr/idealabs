#svPerturb.py
import control as tf
import numpy as np
from scipy import linalg as la
import cmath

"""Functions for calculating a discrete time perturbation of a stable system"""

def svPerturb(G):
    #We fix theta 0 arbitrarily
    theta_0 = np.pi
    SampleT = 1
    
    #Take the svd of G at theta_0
    U,S,Vh = svd(G.evalfr(theta_0))
    sigma = 1/S[0,0]
    u = U[:,1]
    v = Vh.conj().T[:,1]

    ulen = u.shape
    vlen = v.shape

    alpha = complex(1.3, 1.3)


def linkPert(Qij):
    """Perturb a single link
       Param: Qij: single link transfer function object
       Returns: Delta that will perturb that link
    """

    theta_0 = np.pi
    G0 = G.evalfr(theta_0)
    U,S,Vh = la.svd([[G0.real,-G0.imag],[G0.imag,G0.real]])
    sigma = 1/S[0,0]
    u = complex(U[0,0],U[0,1])
    v = complex(Vh[0,0],-Vh[1,0])
    vrho,vphi = cmath.polar(v)
    urho,uphi = cmapth.polar(u)
    
    """Arbitrary Fixing"""
    alpha = complex(1.3,1.3)
    
    beta1 = allPassBeta(theta_0,vphi,alpha)
    beta2 = allPassBeta(theta_0,uphi,alpha)

    utf = tf.tf([-1*beta2.conjugate()*urho,urho],[1,-1*beta2])
    vtf = tf.tf([-1*beta1.conjugate()*vrho,vrho],[1,-1*beta1])

    


def allPassBeta(theta_0,phi,alpha):


    quotient = np.exp(complex(0,1)*phi)*(np.exp(complex(0,1)*theta_0) - alpha)/(1 - np.exp(complex(0,1)*theta_0)*alpha.conjugate())
    a = quotient.real
    b = quotient.imag

    A = np.array([[ a - np.cos(theta_0) , -b - np.sin(theta_0)],
                  [ b - np.sin(theta_0)  , a + np.cos(theta_0)]])
    
    c = np.array([a*np.cos(theta_0) - b*np.sin(theta_0)-1, a*np.sin(theta_0) + b*np.cos(theta_0)])

    x = la.solve(A,c)

    return la.norm(complex(a,b))


def allPassAlpha2(theta,phi):
    A = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    b = np.array([1 - np.cos(.5*(theta+phi)),np.sin(.5*(theta+phi))])
    x = la.solve(A,b)
    return complex(x[0],x[1])
