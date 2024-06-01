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
a = 0
N = 5
om_values = np.linspace(4, 0, 20)
v_values = np.linspace(0, 7, 20)
X, Y = np.meshgrid(om_values, v_values)
Z = np.zeros_like(X)
d0, ini = ising.densidad(N)
A_total =(2*np.pi**2)/0.3**2
# Recorremos los valores de los parametros
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        params = [X[i, j]*gam, Y[i, j]*gam, a, gam]
        H, list_J = ising.ising(params, N)
        L = q.liouvillian(H, list_J)
        todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N +2)
        vals, vects = todoh
        l1 = np.reshape(vects[1], (d0.shape[0], d0.shape[1]))
        posibles, traza = general.buscar_angulos(l1, d0.full(), N)
        Z[i, j] = len(posibles)/A_total
        
"""
# Representamos 
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

ax2.set_title('Nonsense (3 masked regions)')
ax2.set_xlabel('word length anomaly')
ax2.set_ylabel('sentence length anomaly')

# Make a colorbar for the ContourSet returned by the contourf call.
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# Add the contour line levels to the colorbar
cbar.add_lines(CS2)