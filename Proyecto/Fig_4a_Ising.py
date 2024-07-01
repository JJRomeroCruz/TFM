#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 23:16:26 2024

@author: juanjo
"""
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import dicke
import ising
import general
import time
import qutip as q

# Elegimos los parametros queno van a cambiar
gam = 1
a = 1
N = 5
om_values = np.linspace(4, 0.1, 12)
v_values = np.linspace(0.1, 7, 12)
X, Y = np.meshgrid(om_values, v_values)
Z = np.zeros_like(X, dtype = complex)
d0, ini = ising.densidad(N)
salto_theta = 0.1
salto_phi = 0.1
#A_total =(2*np.pi**2)/0.1**2
A_total = 4.0*np.pi
# Recorremos los valores de los parametros
for i in range(X.shape[0]):
    print(i)
    for j in range(X.shape[1]):
        print(j, X[i, j], Y[i, j])
        params = [X[i, j]*gam, Y[i, j]*gam, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        todoh = (L.dag()).eigenstates(sparse = False, sort = 'high', eigvals = N, maxiter = 1e8)
        vals, vects = todoh
        l1 = np.reshape(vects[1], (d0.shape[0], d0.shape[1]))
        posibles, traza = general.buscar_angulos(l1, d0.full(), N)
        suma = 0.0
        for tup in posibles:
            suma += salto_theta*salto_phi*np.sin(tup[0])
        Z[i, j] = suma/A_total
        
        
fig1, ax2 = plt.subplots(layout='constrained')

# Antes he usado viridis
CS = ax2.contourf(X, Y, Z, 10, cmap = 'inferno')

ax2.set_title(r'$a = 1$')
ax2.set_xlabel(r'$V / \gamma$')
ax2.set_ylabel(r'$\Omega / \gamma$')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('Área')

# Add the contour line levels to the colorbar 
plt.savefig('Fig_4a_Ising_a1.png')