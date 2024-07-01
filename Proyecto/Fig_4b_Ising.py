#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:41:57 2024

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
a = 10
N = 5
om_values = np.linspace(4, 0, 12)
v_values = np.linspace(0, 7, 12)
X, Y = np.meshgrid(om_values, v_values)
Z = np.zeros_like(X, dtype = complex)
d0, ini = ising.densidad(N)

# Recorremos los valores de los parametros
for i in range(X.shape[0]):
    print(i)
    for j in range(X.shape[1]):
        print(j)
        params = [X[i, j]*gam, Y[i, j]*gam, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N, maxiter = 1e8)
        vals, vects = todoh
        if(np.isclose(np.imag(vals[2]), 0, atol = 1e-3)):
            Z[i, j] = np.real(vals[1])/np.real(vals[2])
        else:
            Z[i, j] = np.real(vals[1])/np.real(vals[3])
        
# Representamos 
fig1, ax2 = plt.subplots(layout='constrained')
CS = ax2.contourf(X, Y, Z, 10, cmap = 'viridis')

ax2.set_title('a = 10')
ax2.set_xlabel(r'$V / \gamma$')
ax2.set_ylabel(r'$\Omega / \gamma$')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel(r'$\tau_3 / \tau_2$')
# Add the contour line levels to the colorbar
plt.savefig('Fig_4b_Ising_a1.png')