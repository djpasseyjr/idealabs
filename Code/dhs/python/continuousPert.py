#continuousPert.py
import numpy as np
import scipy as sp
from scipy import linalg as la
import numpy.polynomial.polynomial as poly

def pertubation(M,jomega):
    """Given a system matrix M whose norm is greater than
    or equal to one, this function constructs a perturbation
    that destabilizes the system.

    Args: 
        M (2 nxn nd-arrays) : the first matrix is the numerator
                           with norm >= 1 at jomega
        jomega : complex signal
    Returns:
        delta (nxn nd-array) : Destabilizing perturbation
    """

    #Check that the 2 norm of M is not less than one
    if la.norm(M,ord=2)< 1:
        raise ValueError("System Matrix must have a norm greater than 1")

    #for discrete time
#    m,n = M(jomega).shape
#    U,sigma,VH = la.svd(M(jomega))

    m,n = M.shape
    U,sigma,VH = la.svd(M)
    
    #check that M is invertible
    if not (np.round(sigma,7) != 0).all():
        raise ValueError("System matrix must be invertible")

    #Construct delta
    u1 = U[:,0].conj()
    v1 = VH.conj().T[:,0]

    theta = np.tan(u1.imag/u1.real)
    a = abs(u1)
    
    alpha = jomega.real*( 1./np.tan(theta)+np.sqrt((1./np.tan(theta))**2+1) )

    delta = (1./sigma[0])*np.outer(v1,u1)

    return delta

def delta_test():
    singular = True
    
    for n in [2,3,5,10,100]:
        for i in range(3):
            M = np.zeros((n,n))

            while la.norm(M,ord=2) < 1:
                M = .5*np.random.rand(n,n) + .5j*np.random.rand(n,n)

            delta = pertubation(M,0)


            print(n)
            print(la.norm(delta,ord=2))
            assert np.isclose(la.det(np.eye(n) - np.dot(M,delta)), 0)
    
    
