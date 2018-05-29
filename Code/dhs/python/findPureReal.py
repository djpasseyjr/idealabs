#usr/bin/python
import numpy as np
import control as ctrl
from matplotlib import pyplot as plt

"""Functions for creating a simple destabilizing attack on a transfer function"""

def pureReal(G):
    """ Given a transfer function G(z) with real coefficients, return all z0 on the
        complex unit circle such that G(z0) is real. (Scaling z0 by a real number r
        will also result in a real valued G(r*z0).)

        Parameters
        ----------
        G: (tf object) siso transfer function

        Returns
        -------
        realAt: (nd-array) all numbers z0 on the complex unit circle where 
                G(z0) is real
    """
    if G.inputs != 1 or G.outputs != 1:
        raise ValueError("G must be single input-single output")
    if sum(G.num[0][0].imag != 0) != 0 or sum(G.den[0][0].imag != 0) !=0:
        raise ValueError("G must only have real coefficients")


    #Retrieve the numerator and denominator of G(z) and G(1/z)
    P = np.poly1d(G.num[0][0])
    Q = np.poly1d(G.den[0][0])

    Pconj = np.poly1d(conjT(G).num[0][0])
    Qconj = np.poly1d(conjT(G).den[0][0])

    #Find the roots of G(z) - G(1/z)

    topRoots = (P*Qconj - Pconj*Q).r
    bottomRoots = (Q*Qconj).r

    #Remove all places where the roots cancel
    mask = [r not in set(bottomRoots) for r in topRoots]
    realAt = topRoots[mask]

    return realAt

def conjT(siso):
    """Returns the conjugate of a discrete siso transfer function
 
        Parameters 
        ----------
        siso: (tf object) siso transfer function

        Returns
        -------
        siso(-s) if siso is continuous
        siso(1/z) if siso is discrete
        """
    dt = siso.dt
    num = list(siso.num[0][0])
    den = list(siso.den[0][0])

    if dt is None:
        #Multiply the coeffiecients of s^n by (-1)^n in the numerator and denominator.
        num = [(-1**i)*num[i] for i in range(len(num))]
        num[-1] = -1*num[-1] #Correct the coefficient for s^0
        den = [(-1**i)*den[i] for i in range(len(den))]
        den[-1] = den[-1]

    else:
        #Function to find the first nonzero entry of numerator and denominator coefficients
        firstNonZero = lambda x: x.index([num for num in x if num!=0][0])

        #Find the degree of the numerator and denominator
        denDeg = len(den) - firstNonZero(den)
        numDeg = len(num) - firstNonZero(num)
        degDiff = denDeg - numDeg

        #Reverse the order of the coefficients
        den = den[::-1]
        num = num[::-1]

        if degDiff > 0:
            num = num + [0]*degDiff
        
        if degDiff < 0:
            den = den + [0]*(-1*degDiff)

    #take the complex conjugate
    num = [x.conjugate() for x in num]
    den = [x.conjugate() for x in den]

    return ctrl.tf(num,den,dt)

def randomPureReal(N=4,M=3):

    #Generate a random transfer function of degree N

    num = np.poly(2*np.random.rand(N)-1)
    den = np.poly(2*np.random.rand(M)-1)
    G = ctrl.tf(num,den,1)
    realFreq = pureReal(G)
    sizes = [abs(ctrl.evalfr(G,z)) for z in realFreq]
    realFreq = realFreq[np.argsort(sizes)]
    sizes = 10*(np.arange(len(realFreq)) + 1)

    #Plot the unit circle and all the values where it is pure real
    x = np.linspace(0,2*np.pi)
    plt.plot(np.cos(x),np.sin(x))
    plt.scatter(realFreq.real,realFreq.imag,s=sizes,c='r')
    plt.axis('scaled')
    plt.show()
    
