#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 09:52:16 2024

@author: juanjo
"""
import numpy as np
import matplotlib.pyplot as plt
import qutip as q
import dicke
import time
import general
from mpl_toolkits.mplot3d import Axes3D

inicio = time.time()

# Generamos el hamitoniano y el operador de salto de Dicke
N = 10
omega_a = 1
omega = 1.0*omega_a
kappa = 1.0*omega_a
g = 1.0*omega_a
params = [omega_a, omega, kappa, g]

H, J = dicke.dicke_bueno(N, params)
H = q.Qobj(H)
J = q.Qobj(J)
d0, ini = dicke.densidad_bueno(N)
limite_t = 60

# Sacamos el lindbladiano y sus autovalores
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N+3)
vals, vects = todoh

# Sacamos la transformación del efecto Mpemba
U2, U_cambio, vals_l = general.Mpemba2_mejorada_q(L, vects, vals, d0, N, ini)

# Aplicamos la transformacion de Mpemba y la matriz de cambio de base
d01 = q.Qobj(np.dot(U_cambio.full(), np.dot(d0.full(), (U_cambio.dag()).full())))
d02 = q.Qobj(np.dot(U2.full(), np.dot(d0.full(), (U2.dag()).full())))

# Resolvemos el sistem usando Quantum Jump Montecarlo
print('Montecarlo')
nveces = 100
output1, tiempo, fotones1 = general.Resolver_Sistema(d01, H, [J], N, nveces, limite_t)
output2, tiempo, fotones2 = general.Resolver_Sistema(d02, H, [J], N, nveces, limite_t)

# Estado estacionario
est = q.steadystate(q.Qobj(H), [q.Qobj(J)])

# Obtenemos distancia Hilbert Schmdit
vector1, vector2 = [], []
for i in range(len(output1)):
    x1 = output1[i]-est
    x2 = output2[i]-est
    vector1.append(np.sqrt((x1.dag()*x1).tr()))
    vector2.append(np.sqrt((x2.dag()*x2).tr()))
    
fin = time.time()
print('Tiempo: ' + str(fin-inicio))

# Representamos todo
fig = plt.figure(figsize = (12, 12))

# Distancia de Hilbert Schmidt
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(tiempo, vector1, 'b.-', label = 'Normal')
ax1.plot(tiempo, vector2, 'g.-', label = 'Mpemba')

ax1.grid(True)
ax1.legend()

# Overlap
ax2 = fig.add_subplot(3, 1, 2)
s = np.linspace(0, 0.5*np.pi, 1000)
c_s = vals_l[0]*np.cos(s)**2 + vals_l[1]*np.sin(s)**2
ax2.plot(s, c_s)

ax2.set_xlabel('s')
ax2.set_ylabel('c(s)')

# Parte de distribucion de fotones
#ax3 = fig.add_subplot(3, 1, 3)

plt.show()