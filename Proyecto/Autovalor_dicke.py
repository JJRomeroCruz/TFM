#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:02:27 2024

@author: juanjo
"""
""" En este script se estudia la parte imaginaria del autovalor mayor distinto de 0 segun los valores de g """

import numpy as np
import dicke
import general
import matplotlib.pyplot as plt
import qutip as q

# Elegimos los parametros del sistema
N = 40
#sigma = 1.0
#k = 0.1*sigma
#g = 1.0*sigma
#w = 0.01*sigma
sigma = 1.0
k = 1*sigma
#g = 0.1*sigma
g = 0*np.sqrt(2)*sigma
w = 1*sigma
params = [sigma, w, k, g]
g_final = 8*np.sqrt(2)*sigma
vectorg = []
vectorl = []
while(g < g_final):
    # Sacamos el lindbladiano
    params = [sigma, w, k, g]
    H, J = dicke.dicke_bueno(N, params)
    L = q.liouvillian(q.Qobj(H), [q.Qobj(J)])
    
    # Diagonalizamos el lindbladiano
    todo = L.eigenstates(sparse = False, sort = 'high', eigvals= N + 5)
    vects = todo[1]
    vals = todo[0]
    
    # Almacenamos los autovalores
    vectorg.append(g)
    L_buenos= []
    for i in range(len(vals)):
        if(np.allclose(L*vects[i], np.zeros_like(L*vects[i]), atol = 1e-3) == False):
            L_buenos.append(vals[i])
    #vectorl.append(vects[0][-1] - vects[0][-2])
    vectorl.append(L_buenos[0])    
    g += 0.2*sigma
    
# Representamos la grafica
plt.vlines(np.sqrt(2)*sigma, -1, 1)
plt.plot(vectorg, [np.real(elemento) for elemento in vectorl], 'ro')

fig, ax = plt.subplots(1, 2, figsize = (10, 5))

ax[0].plot(vectorg, [np.imag(elemento) for elemento in vectorl], 'ro')
ax[0].set_title('Parte imaginaria')

ax[1].plot(vectorg, [np.real(elemento) for elemento in vectorl], 'bo')
ax[1].set_title('Parte real')