#adjointPert.py
import numpy as np
import control as ctrl
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#The links below are local TODO: link to the appropriate file 
from dsf import inverse_rational_matrix
from dsf import build_identity_tf

def linkPerturb(Q,i,j):
    """Generates a perturbed Q matrix. Creates smallest pertubation 
       capable of destabilizing the link from i to j and adds the perturbation to Q.         
       
       Parameters
       ----------
       Q: (tf object) The Q matrix from the dynamical structure 
                 function of the system
       i: (int) row of Q
       j: (int) column of Q

       Returns
       -------
       Qpert: (tf object) perturbed Q matrix

       Notes
       -----
       This function creates a pertubation P who's norm is minimal such that when P 
       is added to the i,jth entry of Q, the new transfer function is unstable.

    """
    m,n,r = np.shape(Q.num)
    dt = Q.dt
    I = build_identity_tf(m)
    I.dt = dt

    #Calculte the transfer frunction from j to i shown to be the j,ith 
    #entry of(I-Q)^-1
    IminQinv = inverse_rational_matrix(I-Q)
    G = ctrl.tf(IminQinv.num[j][i],IminQinv.den[j][i],dt)

    #Calculate the perturbation    
    delta = adjAttack(G)

    #Add to the transfer function
    Qpert = addToTfLink(Q,i,j,delta)
    return Qpert

def adjAttack(siso):
    """Function for creating a destabilizing pertubation for a given stable
       discrete time siso transfer function.

       Parameters
       ----------
       siso: (tf object) Discrete time siso transfer function

       Returns 
       -------
       D: (tf object) Minimum perturbation to destabilize G
    """
    dt = siso.dt
    sigma = norm(siso)      
    D = conjT(siso)/sigma**2
    
    return D


def fpeak(siso):
    """Estimates the frequency where the transfer function achieves it's 
       peak magnitude.

       Parameters 
       ----------
        siso: (tf object) siso transfer function

       Returns
       -------
       theta: (real number) The number theta such that  |G(exp(i*theta))| 
       is maximized (discrete time)
                or
       jomega: (imaginary number) The number jomega such that |G(jomega)| 
       is maximized (continuous time). 
    """

    dt = siso.dt
    #CONTINUOUS TIME
    if dt is None:
        jomega = np.zeros(100,complex)
        jomega.imag = np.linspace(0,2*np.pi,100)
        gain = np.apply_along_axis(lambda x : ctrl.evalfr(siso,x),0,jomega)
        gain = (gain*gain.conjugate())**.5
        return jomega[np.argmax(gain)]

    #DISCRETE TIME
    else:
        theta = np.linspace(0,2*np.pi,100)
        gain = np.apply_along_axis(lambda x : ctrl.evalfr(siso,np.exp(x*1j)),0,theta)
        gain = (gain*gain.conjugate())**.5
        return theta[np.argmax(gain)]

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

def addToTfLink(G,i,j,siso):
    """ Performs the operation G[i,j] = G[i,j] + siso, on transfer function G

        Parameters
        ----------
         G: (tf object)
         i: (int) row of G
         j: (int) column of G
         siso: (siso tf object)

        Returns
        -------
        Gplus: (tf object) The function G with siso added to the 
        apropriate link
    """
    m,n,r = np.shape(G.num)
    dt = G.dt

    #Create a new transfer fuction Add, that is zero except the i,jth entry    
    Addnum = []
    Addden = []
    rowN = [np.array([0.])]*n
    rowD = [np.array([1.])]*n

    for k in range(m):
        Addnum.append(list(rowN))
        Addden.append(list(rowD))

    #Make the i,jth entry of Add equal to siso
    Addnum[i][j] = siso.num[0][0]
    Addden[i][j] = siso.den[0][0]
    Add = ctrl.tf(Addnum,Addden,dt)

    #Use the control library + function to add the new transfer function to G
    Gplus = G + Add
    return Gplus

def norm(siso):
    """ Estimates the norm of a siso transfer function

        Parameters
        ----------
        siso: (tf object) siso transfer function

        Returns
        -------
        norm: the infinity norm of siso (frequency peak)
    """
    dt = siso.dt
    peak = fpeak(siso)
    
    if dt is None:
        norm = abs(ctrl.evalfr(siso,1j*peak))
    else:
        norm = abs(ctrl.evalfr(siso,np.exp(1j*peak)))
    return norm

#UNUSED FUNCTIONS BELOW
def inv(siso):
    """Returns the inverse of a discrete siso transfer function
        
        Parameters 
        ----------
        siso: (tf object) siso transfer function

        Returns
        -------
        the inverse of siso
"""
    num = siso.num[0][0]
    den = siso.den[0][0]
    return ctrl.tf(den,num,siso.dt)

def pertPole(siso):
    """Solves for the poles of the transfer function
        
        Parameters
        ----------
        siso: (tf object): siso transfer function object

        Returns
        ------- 
        p: (ndarray of complex numbers): poles of inv(I-G*attack(G))
        normP: (ndarray) their respective norms
    """
    p = np.roots(inv(ctrl.tf(1,1,1)-siso*adjAttack(siso)).den[0][0])
    normP = (p*p.conjugate())**.5
    return p, normP


"""TESTING"""

def graphPoleSize(N=30):
    x = np.linspace(-1,1,N)
    X,Y = np.meshgrid(x,x)
    Z = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            if (((X[i,j]**2)+(Y[i,j])**2)**.5).real <= 1:
                G = ctrl.tf(1.,[1.,X[i,j]+Y[i,j]*1j],1.)
                Z[i,j] = max(pertPole(G)[1]).real
            else:
                Z[i,j] = 1
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
    plt.show()

def testRandomPole(N=200):
    maxP = np.array([.1]*N);
    for i in range(N):
        r = np.random.rand(4)*2**.5 - (2**.5)/2
        G = ctrl.tf([1.,r[0]+r[1]*1j],[1.,r[2]+r[3]*1j],1)
        maxP[i] = max(pertPole(G)[1].real)

    return maxP
