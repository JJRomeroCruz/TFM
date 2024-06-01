#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 00:53:12 2024

@author: juanjo
"""
import numpy as np
import random as random
import dicke 
import sympy as sp
import matplotlib.pyplot as plt
import general
import qutip as q
import rydberg
import time
import ising

inicio = time.time()

# Parámetros del sistema
N = 40 # Número de átomos
omega = 1 # Frecuencia de transición de los átomos
delta = -1.0*omega  # Frecuencia del modo del campo electromagnético
v = 3.0*omega  # Constante de acoplamiento átomo-modo
kappa = 1*omega
params = [omega, delta, v, kappa]


# Hamiltoniano del sistema y operador de salto
H, J = rydberg.rydberg(N, params)

d0, ini = dicke.densidad_bueno(N)
#d0, ini = rydberg.densidad(N)
psi0 = ini
print(H.shape)
#d0 = Qobj(d0)

# Lista de tiempos de la simulación
tlist = np.linspace(0, 500, 5000)

# Sacamos el lindbladiano y sus autovalores
L = q.liouvillian(H, [J])
todoh = (L.dag()).eigenstates(sparse = True, sort = 'high', eigvals = N + 20)
todo = L.eigenstates(sparse = True, sort = 'high', eigvals = N + 20)
vals = todoh[0]
vects = todoh[1]
r = [q.Qobj(np.reshape(x, (N+1, N+1))) for x in todo[1]]
l = [q.Qobj(np.reshape(x, (N+1, N+1))) for x in todoh[1]]


# Efecto Mpemba
U1 = general.Mpemba1_mejorada_q(L, vects, vals, d0, N, ini)[0]
U2, U_cambio, vals_l = general.Mpemba2_mejorada_q(L, vects, vals, d0, N, ini)
U_cambio = q.Qobj(U_cambio)

ini1 = np.dot(U1.full(), ini.full())
ini2 = np.dot(U2.full(), ini.full())
d01 = np.dot(np.dot(U1.full(), d0.full()), (U1.dag()).full())
d02 = np.dot(np.dot(U2.full(), d0.full()), (U2.dag()).full())
print(H.shape)


# Integramos la ecuación maestra
options = q.Options(nsteps = 1e4)
output1 = q.mesolve(H, U_cambio*q.Qobj(d0)*U_cambio.dag(), tlist, [J], options=options)
output2 = q.mesolve(q.Qobj(H.full()), q.Qobj(d01), tlist, [q.Qobj(J.full())], options=options)
output3 = q.mesolve(q.Qobj(H.full()), q.Qobj(d02), tlist, [q.Qobj(J.full())], options=options)

"""
output1 = general.solucion(d0, r, l, vals, tlist)
output2 = general.solucion(q.Qobj(d01), r, l, vals, tlist)
output3 = general.solucion(q.Qobj(d02), r, l, vals, tlist)
"""
est = q.steadystate(q.Qobj(H), [q.Qobj(J)])
vector1, vector2, vector3 = [], [], []


for i in range(len(output1.states)):
    x1 = output1.states[i] - est
    x2 = output2.states[i] - q.Qobj(est.full())
    x3 = output3.states[i] - q.Qobj(est.full())
    vector1.append(np.sqrt(np.trace(x1.conj()*x1)))
    vector2.append(np.sqrt(np.trace(x2.conj()*x2)))
    vector3.append(np.sqrt(np.trace(x3.conj()*x3)))

"""
for i in range(len(output1)):
    x1 = q.Qobj(output1[i]) - est
    x2 = q.Qobj(output2[i]) - est
    x3 = q.Qobj(output3[i]) - est
    vector1.append(np.sqrt(np.trace(x1.dag()*x1)))
    vector2.append(np.sqrt(np.trace(x2.dag()*x2)))
    vector3.append(np.sqrt(np.trace(x3.dag()*x3)))
    #vector1.append(np.sqrt(np.trace(x1**2)))
    #vector2.append(np.sqrt(np.trace(x2**2)))
    #vector3.append(np.sqrt(np.trace(x3**2)))
"""
fin = time.time()
print('Tiempo: ' + str(fin-inicio))
"""
plt.plot(tlist, vector1, 'b.-', label = 'Normal')
plt.plot(tlist, vector2, 'r.-', label = 'Mpemba 1')
plt.plot(tlist, vector3, 'go', label = 'Mpemba 2')
plt.plot(tlist, vector4, 'y.-', label = 'Mpemba_ang')
plt.legend()
plt.title('g = ' + str(g))
plt.show()

# Generamos la grafica de las posibles transformaciones individuales
radio = 1.0
theta_rango = (0, np.pi)
phi_rango = (0, 2*np.pi)


posibles_filtro, traza_filtro = [], []
for i in range(len(traza)):
    if(traza[i] < 5e-2):
        traza_filtro.append(traza[i])
        posibles_filtro.append(posibles[i])

theta_values = [tup[0] for tup in posibles_filtro]
phi_values = [tup[1] for tup in posibles_filtro]

x1, y1, z1 = general.parametrizacion_esfera(radio, theta_rango, phi_rango)
x2, y2, z2 = general.esfera_partes(radio, theta_values, phi_values)
"""
fig = plt.figure(figsize = (12, 12))

# Parte distancias
ax1 = fig.add_subplot(3, 1, 1)
ax1.plot(tlist, vector1, 'b.-', label = 'Normal')
#ax1.plot(tlist, vector2, 'r.-', label = 'Mpemba 1')
ax1.plot(tlist, vector3, 'g.-', label = 'Mpemba')

ax1.grid(True)
ax1.legend()

# Parte esfera
ax2 = fig.add_subplot(3, 1, 2)

s = np.linspace(0, np.pi*0.5, 1000)
c_s = vals_l[0]*np.cos(s)**2 + vals_l[1]*np.sin(s)**2
ax2.plot(s, c_s)
ax2.plot(s, np.zeros_like(c_s))

ax2.set_xlabel('s')
ax2.set_ylabel('c(s)')

"""
# Parte valor esperado del hamiltoniano respecto del tiempo
ax3 = fig.add_subplot(3, 1, 3)
expect_H1 = q.expect(q.Qobj(H), output1.states)
expect_H2 = q.expect(q.Qobj(H.full()), [q.Qobj(elemento.full()) for elemento in output2.states])
expect_H3 = q.expect(q.Qobj(H.full()), [q.Qobj(elemento.full()) for elemento in output3.states])

#print(expect_H1)

ax3.plot(tlist, expect_H1, label = 'Normal')
ax3.plot(tlist, expect_H2, label = 'Mpemba1')
#ax3.plot(tlist, expect_H3, label = 'Mpemba2')
ax3.legend()
"""
plt.show()