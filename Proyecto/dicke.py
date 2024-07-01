#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 09:45:59 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import scipy as sc
import random as random
import matplotlib.pyplot as plt
import qutip as q
import general


# Funcion que me devuelve el operador de espin en una cierta componente y el operador de spin total
def spin_operator(N, component):
    """
    N: int (numero de qubits)
    Component: str (componente del momento angular, 'x', 'y', 'z')
    returns Qobj
    """
    if component == 'x':
        op = q.sigmax()
    elif component == 'y':
        op = q.sigmay()
    elif component == 'z':
        op = q.sigmaz()
    else: 
        raise ValueError('La componente tiene que se x, y, o z')
    
    # Construimos los operadores
    total_operator = 0
    indv_operators = []
    for i in range(N):
        operators = [q.qeye(2) for _ in range(N)]
        operators[i] = op
        indv_operators.append(q.tensor(operators))
        total_operator += q.tensor(operators)
    
    return total_operator, indv_operators

def dicke_bueno(N, params):
    M = 2
    #M = 8
    j = N/2.0
    n = 2*j + 1
    Jz = q.jmat(j, 'z')
    Jx = q.jmat(j, 'x')
    #Jz = q.tensor(q.qeye(M), q.jmat(j, 'z'))
    #Jx = q.tensor(q.qeye(M), q.jmat(j, 'x'))
    #Jz = q.tensor(q.qeye(M), spin_operator(N, 'z')[0])
    #Jx = q.tensor(q.qeye(M), spin_operator(N, 'x')[0])
    
    sigma = params[0]
    w = params[1]
    k = params[2]
    g = params[3]
    
    
    H = sigma*Jz - ((4.0*w*g*g)/(4.0*w**2 + k**2))*(1.0/N)*Jx*Jx
    J = ((2.0*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*Jx
    #J = -((np.sqrt(k)*g*(4*w + 2*1.j*k))/(np.sqrt(N)*(4*w*w + k*k)))*Jx
    return H, J

def densidad_bueno(N):
    M = 2 # Numero de fotones
    j = N/2.0
    n = 2*j + 1
    Jz = q.jmat(j, 'z')
    print(Jz.shape)
    # Creamos la base de autoestados de sz
    todo = Jz.eigenstates(sparse = True, sort = 'high')
    base = todo[1]

    # Construimos el vector inicial
    a, b = random.random(), random.random()
    ini = (a + b*1.j)*base[0]
    for i in range(1, len(base)):
      a, b = random.random(), random.random()
      ini += (a + b*1.j)*base[i]
    # Le hacemos el ketbra para construir la matriz densidad

    ini = ini.unit()
    d = ini*ini.dag()
    return d, ini