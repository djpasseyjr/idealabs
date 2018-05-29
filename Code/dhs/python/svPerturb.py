#svPerturb.py
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
    delta = sisoAttack(G)

    #Add to the transfer function
    Qpert = addToTfLink(Q,i,j,delta)
    return Qpert

def sisoAttack(siso):
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

    #Find the frequency peak and take a specific SVD at that frequency
    theta = fpeak(siso)
    s0 = ctrl.evalfr(siso,np.exp(theta*1j))
    sigma,phi = abs(s0),np.angle(s0)
    #The numbers, u = np.exp(phi*1j), v=1 complete the SVD but are not needed

    #CONTINUOUS TIME
    if dt is None:
        sign = 1

        #Check the angle in order to ensure a solution
        if phi > 0:
            phi = phi - np.pi
            sign = -1

        #Calculate all pass filter parameter
        jomega = theta
        uH = np.exp(-phi*1j)
        a = jomega*(1 - uH)/(1 + uH)
        utf = ctrl.tf([1.,-a],[1., a])
        D = utf/sigma

    #DISCRETE TIME
    else:

        #Find the frequency peak and take a specific SVD at that frequency
        theta = fpeak(siso)
        s0 = ctrl.evalfr(siso,np.exp(theta*1j))
        sigma,phi = abs(s0),np.angle(s0)
        #The numbers, u = np.exp(phi*1j), v=1 complete the SVD but are not needed

        #Solve for all pass filter parameter
        a = np.exp(-1j*theta) - np.exp(-.5j*(phi+theta))
        
        #Choose appropriate filter
        if ((a*a.conjugate())**.5).real < 1:
            utf = ctrl.tf([1.,-1.*a.conjugate()],[-1.*a,1.],1.)
        else:
            utf = np.exp(2j*phi)*ctrl.tf([-1.*a,1.],[1.,-1.*a.conjugate()],1.)

        #Create a unitary transfer function and incoorperate it into D
        D = conj(utf)/sigma
        
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

def conj(siso):
    """Returns the conjugate of a discrete siso transfer function
 
        Parameters 
        ----------
        siso: (tf object) siso transfer function

        Returns
        -------
        the conjugate of siso
        """
    num = siso.num[0][0][::-1].conjugate()
    den = siso.den[0][0][::-1].conjugate()
    return ctrl.tf(num,den,siso.dt)

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
    p = np.roots(inv(ctrl.tf(1,1,1)-siso*sisoAttack(siso)).den[0][0])
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
