import math
import numpy as np
import scipy
from scipy.linalg import eig
from scipy.linalg import eigh
from matrix import*
import matrix
from pedestrian import*
from sympy import *

import pedestrian


def indicat(x, lb, numped):
    """"
    Indicator function which checks whether the pedestrian is or not on the bridge, it
    returns 1 if it is on the bridge, otherwise zero.
    Parameters
    ----------
    x : vector shows the positions of wheels
        Unit m.
    lb : single span length
        Unit m.
    numped : number of pedestrians
       
        Returns
    -------
    None
    """
    I = np.zeros((numped))  # important
    for i in range(numped):
        if 0 < x[i] < lb:
            I[i] = 1
        else:
            I[i] = 0
    return I

def Phi_x(x_interest, lb, rho, N_bridge):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x_interest : interest position at the bridge.
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
    # the end
    n = N_bridge
    phi_x = np.zeros((n, 1))
    for j in range(n):
        jt = j + 1
        phi_x[j] = (2 / rho / lb) ** 0.5 * sin(jt * pi * x_interest / lb)
    return phi_x




def Phi_matrix(xrb, lb, rho, N_bridge,numped):
    """"
    derive the mode shape of interested position at the bridge.
    Parameters
    ----------
    x : position of pedestrians with time
        Unit m.
    lb : single span length
        Unit m.
    rho : unit density of bridge
        Unit kg/m.
    N_bridge : number of modes taken into account.
        
    Returns
    -------
    None
    """
   
    n = N_bridge
    N = np.zeros((n, numped))
    for i in range(n):
        for j in range(numped):
          it = i + 1
          N[i][j] = (2 / rho / lb) ** 0.5 * sin(it * pi * xrb[j] / lb) #x interest should be time dependednt x=vt xrb is a matrix with time varying location of each pedestrian
    I = indicat(xrb, lb, numped) 
    NN = np.dot(N, np.diag(I))
    return NN
    

def MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,t):
    """assembles the matrices with coupled SMD properties. Coupling is in M here"""
    #mped kped cped are the individual property containing matrices

    mass=bridge.Mass_matrix(self=case_bridge)
    k= bridge.Stiffness_matrix(self=case_bridge)
    c=bridge.Damp_matrix(self=case_bridge)
    NN = Phi_matrix(xrb, lb, rho, N_bridge,numped)
    v=Pedestrian.detVelocity
    
    #M assemble
    M3 =np.diag(mped)
    M1= np.hstack((mass,NN*M3)) 
    M2= np.zeros((numped,N_bridge))#(diag(mped).shape)
    M4 = np.hstack((M2,M3))
    M = np.vstack((M1,M4))
    #end

    #K assemble
    K2= np.zeros((N_bridge,numped))#(diag(k).shape)
    K3= np.hstack((k,K2))
    K4 = np.diag(kped)
    K5=np.hstack((-np.array(NN*K4).T,K4))
    K=np.vstack((K3,K5))
    #end

    #C assemble
    C2= np.zeros((N_bridge,numped))
    C3= np.hstack((c,C2))
    C4 = np.diag(cped)
    C5=np.hstack((-np.array(NN*C4).T,C4))
    C=np.vstack((C3,C5))
    #end
    
    #F matrix
    Ft = pedestrian.calcPedForce(case_pedestrian, t)  #this should get a set of arrays and choose the value for corresponding xrb
    #F = np.vstack((np.sum(NN * Ft, axis=1).reshape(-1, 1), np.zeros((numped, 1))))
    
    #F = np.vstack((np.sum(NN*Ft, axis=1),zeros(numped,1)))
    
    #end
    return M,K,C,Ft


def Newmarksuper_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h

    #Mc = np.zeros((N_bridge+numped,N_bridge+numped))
    #Mc[0:N_bridge, 0:N_bridge] = bridge.Mass_matrix()

    Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    #print(Mc)
    #print(Kc)
    #print(Cc)
    #print(Fc)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)
    #print(Fc)
    Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
        #print(xr) # new location of pedestrians where v is an array of pedestrian velocities
        #xr = np.concatenate(xr)  # important
        # print('...',it,n-1,xr,np.shape(xr))
        Mc, Kc, Cc, Fc = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*Fc, np.zeros((numped, 1))))
        #print(Mc)
        #print(Kc)
        #print(Cc)
        #print(NN)
        #print(Fc)
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu
    

def accdyn_super(bridge_instance,ddu, x_inter, hht):
    """
    Generate the acceleration vector (time) of the interested point
    at the bridge.
    Parameters
    ----------
    ddu : acceleration of the HSI system.
        Unit m/s^2.
    
    x_inter : dynamic of interest point at the bridge.
        Unit m.
    N_span : number of spans in the bridge.
        1 means single span
    v : pedestrian speed
        m/s.
    hht : time steps
        Unit s.
    -------
    None.
    """
    lb = bridge_instance.L
    rho = bridge_instance.rho
    N_bridge = bridge_instance.n
    v=Pedestrian.detVelocity

    phi_x = Phi_x(x_inter, lb, rho, N_bridge)
   
    #nn = np.zeros(N_bridge)
    #t = np.transpose(np.arange(0, (lb+1) / v, hht))
    #for j in range(N_bridge):
        #nn[j] = 1
    meta = (np.diag(phi_x.flatten()).dot(ddu[:N_bridge, :]))
    
    #meta = ( diag(Phi_x).dot(ddu[:N_bridge, :]))
    column_sums = np.sum(meta, axis=0) 
    #print(column_sums)  
    return column_sums





'''for pseudo excitation method'''

def Newmarkpseudo_HSI(case_pedestrian,case_bridge,numped,N_bridge, lb, hht, v, mped,kped,cped,xrb, rho,force):
    """"
    Solve the coupled HSI matrices by Newmark-beta method.
    Parameters
   
    """

    t = np.transpose(np.arange(0, (lb + 1) / v, hht)) # the last data in the time matrix should be the time that the last pedestrian leaves the bridge
    u0 = np.zeros((numped + N_bridge, 1))
    du0 = np.zeros((numped + N_bridge, 1))
    ######################
    gamma = 1 / 2
    beta = 1 / 4
   
    n = np.size(t)
    h = hht

    # Constant terms and effective stiffness
    a0 = 1 / (beta * h**2)
    a1 = gamma / (beta * h)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1
    a5 = h * (gamma / (2 * beta) - 1)
    a6 = h * (1 - gamma)
    a7 = gamma * h


    Mc, Kc, Cc, _ = MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xrb, lb, rho, N_bridge,numped,0)
    # ddu0 = np.linalg.inv(Mc).dot(Fc - Cc.dot(du0) - Kc.dot(u0)) # same as inv(M)*(.)
    
    NN=Phi_matrix(xrb,lb,rho,N_bridge,numped)
    #print(Fc)
    
    Fc=np.vstack((NN*0, np.zeros((numped, 1))))
    ddu0 = np.linalg.solve(Mc, Fc - Cc.dot(du0) - Kc.dot(u0))
    u = np.zeros((numped+N_bridge, n))
    du = np.zeros((numped+N_bridge, n))
    ddu = np.zeros((numped+N_bridge, n))

    u[:, [0]] = u0
    du[:, [0]] = du0
    ddu[:, [0]] = ddu0
    xr = xrb
    for i in range(n-2):
        it = i + 1
        xr = np.add(xr, v * h) 
        #print(xr) # new location of pedestrians where v is an array of pedestrian velocities
        #xr = np.concatenate(xr)  # important
        # print('...',it,n-1,xr,np.shape(xr))
        Mc, Kc, Cc,_= MatrixAssemble(case_pedestrian,case_bridge,mped,kped,cped,xr, lb, rho, N_bridge,numped,t[i])
        NN=Phi_matrix(xr,lb,rho,N_bridge,numped)
        #print(Fc[it])
        Fc=np.vstack((NN*force[:,[i]], np.zeros((numped, 1))))
        #print(Mc)
        #print(Kc)
        #print(Cc)
        #print(NN)
        #print(Fc)
        Feff = (
            Fc
            + Mc.dot(a0 * u[:, [i]] + a2 * du[:, [i]] + a3 * ddu[:, [i]])
            + Cc.dot(a1 * u[:, [i]] + a4 * du[:, [i]] + a5 * ddu[:, [i]])
        )
        Keff = Kc + a0 * Mc + a1 * Cc
        # u[:,[it]]=np.linalg.inv(Keff).dot(Feff)
        u[:, [it]] = np.linalg.solve(Keff, Feff)
        ddu[:, [it]] = (
            a0 * (u[:, [it]] - u[:, [i]]) - a2 * du[:, [i]] - a3 * ddu[:, [i]]
        )
        du[:, [it]] = du[:, [i]] + a6 * ddu[:, [i]] + a7 * ddu[:, [it]]

    print(ddu)
    return u, du, ddu



