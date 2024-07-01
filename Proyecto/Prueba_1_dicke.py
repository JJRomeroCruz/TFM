#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:28:54 2024

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

# Parámetros del sistema
N = 40 # Número de átomos
omega_a = 1 # Frecuencia de transición de los átomos
omega = 1.0*omega_a  # Frecuencia del modo del campo electromagnético
g = 1.0*omega_a  # Constante de acoplamiento átomo-modo
#g = np.sqrt(16)*np.sqrt(omega_a)
kappa = 1*omega_a
params = [omega_a, omega, kappa, g]

# Hamiltoniano del sistema
H, J = dicke.dicke_bueno(N, params)

# Estado inicial
d0, ini = dicke.densidad_bueno(N)
psi0 = ini
print(H.shape)

# Lista de tiempos de la simulación
tlist = np.linspace(0, 20, 80)

# Sacamos el lindbladiano y sus autovalores
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N+3)

vals = todoh[0]
vects = todoh[1]

# Efecto Mpemba
U1 = general.Mpemba1_mejorada_q(L, vects, vals, d0, N, ini)[0]
U2, U_cambio, vals_l = general.Mpemba2_mejorada_q(L, vects, vals, d0, N, ini)

# Resolver la ecuación maestra para obtener la evolución temporal del sistema
print(d0.shape)
ini1 = np.dot(U1.full(), ini.full())
ini2 = np.dot(U2.full(), ini.full())
d01 = np.dot(np.dot(U1.full(), d0.full()), (U1.dag()).full())
d02 = np.dot(np.dot(U2.full(), d0.full()), (U2.dag()).full())
print(ini1.shape == ini.shape)
print(H.shape)
output1 = q.mesolve(H, q.Qobj(ini), tlist, [J])
output2 = q.mesolve(q.Qobj(H.full()), q.Qobj(d01), tlist, [q.Qobj(J.full())])
output3 = q.mesolve(q.Qobj(H.full()), q.Qobj(d02), tlist, [q.Qobj(J.full())])

est = q.steadystate(q.Qobj(H), [q.Qobj(J)])
vector1, vector2, vector3, vector4 = [], [], [], []
for i in range(len(output1.states)):
    x1 = output1.states[i] - est
    x2 = output2.states[i] - q.Qobj(est.full())
    x3 = output3.states[i] - q.Qobj(est.full())
    vector1.append(np.sqrt(np.trace(x1*x1)))
    vector2.append(np.sqrt(np.trace(x2*x2)))
    vector3.append(np.sqrt(np.trace(x3*x3)))


fin = time.time()
print('Tiempo: ' + str(fin-inicio))

fig = plt.figure(figsize = (12, 12))

# Parte distancias
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(tlist, vector1, 'b.-', label = 'Normal')
ax1.plot(tlist, vector2, 'r.-', label = 'Mpemba 1')
ax1.plot(tlist, vector3, 'g.-', label = 'Mpemba 2')

ax1.grid(True)
ax1.legend()
#ax1.set_title('Distancia de H-S. Dicke. g = ' + str(g))

# Parte esfera
ax2 = fig.add_subplot(3, 1, 2)

s = np.linspace(0, np.pi*0.5, 1000)
c_s = vals_l[0]*np.cos(s)**2 + vals_l[1]*np.sin(s)**2
ax2.plot(s, c_s)
ax2.plot(s, np.zeros_like(c_s))

ax2.set_xlabel('s')
ax2.set_ylabel('c(s)')

plt.show()