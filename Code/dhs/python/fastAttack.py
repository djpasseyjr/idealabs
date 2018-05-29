#!/usr/bin/python
import numpy as np
import control as ctrl
import adjointPert as adj
from dsf import inverse_rational_matrix

def linkPerturb(Q,i,j):
    """Generates a perturbed Q matrix. Creates a pertubation 
       capable of destabilizing the link from i to j and adds the perturbation to Qij.         
       
       Parameters
       ----------
       Q: (tf object) The Q matrix from the dynamical structure 
                 function of the system
       i: (int) row of Q
       j: (int) column of Q

       Returns
       -------
       Qpert: (tf object) perturbed Q matrix

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
    delta = attackLink(G)

    #Add to the transfer function
    Qpert = addToTfLink(Q,i,j,delta)
    return Qpert

def attackLink(G):
    """ Function for creating a destabilizing pertubation for a given stable
       discrete time siso transfer function.

       Parameters
       ----------
       G: (tf object) Discrete time siso transfer function

       Returns 
       -------
       D: (tf object) Perturbation (Not minimal) to destabilize G
    """
    dt = G.dt
    neg = ctrl.evalfr(G,-1.)
    pos = ctrl.evalfr(G,1.)

    if abs(neg) > abs(pos):
        D = -1*ctrl.tf(1,[1,0],dt)/neg

    else:
        D = ctrl.tf(1,[1,0],dt)/pos

    return D

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


