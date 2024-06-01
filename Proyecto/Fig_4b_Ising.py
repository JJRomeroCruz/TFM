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
a = 0
N = 5
om_values = np.linspace(4, 0, 30)
v_values = np.linspace(0, 7, 50)
X, Y = np.meshgrid(om_values, v_values)
Z = np.zeros_like(X)
d0, ini = ising.densidad(N)
A_total =(2*np.pi**2)/(0.3**2)
# Recorremos los valores de los parametros
for i in range(X.shape[0]):
    print(i)
    for j in range(X.shape[1]):
        params = [X[i, j]*gam, Y[i, j]*gam, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N +2)
        vals, vects = todoh
        Z[i, j] = np.real(vals[1])/np.real(vals[2])
        
# Representamos 
"""
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
#ax.plot_surface(X, Y, Z, cmap='inferno')
ax.contour(X, Y, Z)
ax.set_xlabel(r'$\Omega / \gamma$')
ax.set_ylabel(r'$v / \gamma$')
#ax.set_zlabel('Energía')
plt.title('Energy Landscape del Modelo de Dicke')
plt.show()
"""
fig1, ax2 = plt.subplots(layout='constrained')
#CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone)
CS = ax2.contourf(X, Y, Z, 10, cmap = 'viridis')
# Note that in the following, we explicitly pass in a subset of the contour
# levels used for the filled contours.  Alternatively, we could pass in
# additional levels to provide extra resolution, or leave out the *levels*
# keyword argument to use all of the original levels.

#CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')

ax2.set_title('a = 0')
ax2.set_xlabel(r'$V / \gamma$')
ax2.set_ylabel(r'$\Omega / \gamma$')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel(r'$\tau_3 / \tau_2$')
# Add the contour line levels to the colorbar
#cbar.add_lines(CS2)