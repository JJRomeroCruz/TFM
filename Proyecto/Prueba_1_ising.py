#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:33:09 2024

@author: juanjo
"""
import ising
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import general
import dicke
import qutip as q

# Parametros
N = 5
om = 3
v = 2.5
#a = 1
a = 0
gam = 1
params = [om, v, a, gam]

# Hamiltoniano y operadores de salto
H, list_J = ising.ising(params, N)

# Matriz densidad inicial y tiempo
d01, ini = ising.densidad(N)
d02 = q.basis(int(2**N), 1)*(q.basis(int(2**N), 1)).dag()
lamb = 1
d0 = lamb*d01 + (1-lamb)*d02
tlist = np.linspace(0, 5, 100)

# Lindbladiano
L = q.liouvillian(H, list_J)

# Diagonalizamos el lindbladiano y obtenemos sus autovectores izqu
todoh = (L.dag()).eigenstates(sparse = False, sort = 'high', eigvals = N + 2)
vals, vects = todoh

# Sacamos Mpemba
l1 = np.reshape(vects[1], (d0.shape[0], d0.shape[1]))
posibles, traza = general.buscar_angulos(l1, d0.full(), N)
theta, phi = posibles[traza.index(min(traza))]
U3 = general.Mpemba_sep(theta, phi, N)

# Sacamos la nueva matriz
d03 = np.dot(np.dot(U3.full(), d0), (U3.dag()).full())

# Resolvemos la ecuacion maestra
output1 = q.mesolve(q.Qobj(H.full()), q.Qobj(d0.full()), tlist, [q.Qobj(elemento.full()) for elemento in list_J])
output2 = q.mesolve(q.Qobj(H.full()), q.Qobj(d03), tlist, [q.Qobj(elemento.full()) for elemento in list_J])

# Representamos la esfera de las posibles transformaciones y la distancia de hilbert schmidt
est = q.steadystate(q.Qobj(H.full()), [q.Qobj(elemento.full()) for elemento in list_J])
vector1, vector2 = [], []
for i in range(len(output1.states)):
    x1 = output1.states[i] - est
    x2 = output2.states[i] - q.Qobj(est.full())
    vector1.append(np.sqrt(np.trace(x1.dag()*x1)))
    vector2.append(np.sqrt(np.trace(x2.dag()*x2)))

fig = plt.figure(figsize = (12, 12))

# Parte distancias
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(tlist, vector1, 'b.-', label = 'Normal')
ax1.plot(tlist, vector2, 'r.-', label = 'Mpemba')

ax1.grid(True)
ax1.legend()
ax1.set_title('Distancia H-S. N = ' + str(N))

radio = 1.0
theta_rango = (0, np.pi)
phi_rango = (0, 2*np.pi)


posibles_filtro, traza_filtro = [], []
for i in range(len(traza)):
    if(traza[i] < 1e-2):
        traza_filtro.append(traza[i])
        posibles_filtro.append(posibles[i])

theta_values = [tup[0] for tup in posibles_filtro]
phi_values = [tup[1] for tup in posibles_filtro]

x1, y1, z1 = general.parametrizacion_esfera(radio, theta_rango, phi_rango)
x2, y2, z2 = general.esfera_partes(radio, theta_values, phi_values)

fig = plt.figure()
ax2 = fig.add_subplot(projection = '3d')
ax2.plot_surface(x1, y1, z1, color = 'b', alpha = 0.3)
ax2.scatter(x2, y2, z2, c = 'r', marker = 'o', label = 'Puntos')
ax2.set_aspect('auto')
ax2.set_title('Posibles rotaciones')

plt.show()