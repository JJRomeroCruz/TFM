#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:27:02 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import qutip as q
import matplotlib.pyplot as plt

# Funcion que me genera el hamiltoniano del modelo de Dicke con qutip
def dicke(N, params):
    # Operadores
    sx = 0.5*q.sigmax()
    sz = 0.5*q.sigmaz()
    id_N = q.identity(N)
    
    jx = q.tensor([sx for i in range(N)])
    jz = q.tensor([sz for i in range(N)])
    
    # Parametros
    omega, w, g, k = params
    
    H = omega*jz - ((4*w*g*g)/((4*w*w + k*k))*N)*jx*jx
    J = ((2*np.abs(g)*np.sqrt(k))/(np.sqrt(N*(4*w*w + k*k))))*jx
    
    return H, J

