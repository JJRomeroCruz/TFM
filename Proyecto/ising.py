#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 18:35:42 2024

@author: juanjo
"""
import numpy as np
import general
import qutip as q
import sympy as sp
import matplotlib.pyplot as plt
import random as random
import dicke

# Funcion que me da el hamiltoniano de Ising cuantico y su operador de salto
def ising(params, N):
    # Parametros
    om = params[0]
    v = params[1]
    a = params[2]
    gam = params[3]
    
    # Sacamos los terminos
    spin_z = dicke.spin_operator(N, component = 'z')[1]
    todo_x = dicke.spin_operator(N, component = 'x')[0]
    spin_x = dicke.spin_operator(N, component = 'x')[1]
    spin_y = dicke.spin_operator(N, component = 'y')[1]
    
    list_J = []
    suma = 0
    for i in range(len(spin_z)):
        list_J.append((np.sqrt(gam)/2)*(spin_x[i] - 1.j*spin_y[i]))
        for j in range(len(spin_z)):
            if(i < j):
                suma += (spin_z[i]*spin_z[j])/(np.abs(i-j)**a)
                
    H = om*todo_x + v*suma
    return H, list_J

# Funcion que me permite obtener la matriz densidad de la que hablan en el articulo
def densidad(N):
    return q.basis(int(2**N), 0)*(q.basis(int(2**N), 0)).dag(), q.basis(int(2**N), 0)

# Funcion que me da el hamiltoniano y el operador de salto del modelo de Ising pero para a tendiendo a infinito
def ising_inf(params, N):
    # Parametros
    om = params[0]
    v = params[1]
    a = params[2]
    gam = params[3]
    
    # Sacamos los terminos
    spin_z = dicke.spin_operator(N, component = 'z')[1]
    todo_x = dicke.spin_operator(N, component = 'x')[0]
    spin_x = dicke.spin_operator(N, component = 'x')[1]
    spin_y = dicke.spin_operator(N, component = 'y')[1]
    
    list_J = []
    suma = 0
    for i in range(len(spin_z)):
        list_J.append((np.sqrt(gam)/2)*(spin_x[i] - 1.j*spin_y[i]))
        for j in range(len(spin_z)):
            if(i < j):
                suma += spin_z[i]*spin_z[j]/np.abs(i-j)**a
                
    H = om*todo_x
    return H, list_J