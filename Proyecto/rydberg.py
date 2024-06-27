#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:53:06 2024

@author: juanjo
"""
import numpy as np
import random as random
import dicke 
import sympy as sp
import matplotlib.pyplot as plt
import general
import qutip as q

# Funcion que me obtiene el hamiltoniano del modelo de atomos de rydberg
def rydberg(N, params):
    omega = params[0]
    delta = params[1]
    v = params[2]
    k = params[3]
    j = N/2
    
    Sx = q.jmat(j, 'x')
    Sz = q.jmat(j, 'z')
    Sm = q.jmat(j, '-')
    
    H = omega*Sx - delta*Sz + (v/N)*Sz*Sz
    J = np.sqrt(k/N)*Sm
    
    return H, J

def densidad(N):
    return q.basis(int(N+1), 0)*(q.basis(int(N+1), 0)).dag(), q.basis(int(N+1), 0)


